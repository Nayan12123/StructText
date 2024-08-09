""" entity linking at segment-level"""
import json
import paddle as P
import numpy as np
from paddle import nn
from model.encoder import Encoder
from paddle.nn import functional as F
from model.ernie.modeling_ernie import ACT_DICT, append_name, _build_linear, _build_ln

class CombinedLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.margin = margin

    def forward(self, Pi, Pj, y, targets):
        """
        Arguments:
        - Pi: Tensor of predicted probabilities for positive samples (batch_size, )
        - Pj: Tensor of predicted probabilities for negative samples (batch_size, )
        - y: Tensor of ground truth labels (-1 for Pi < Pj, 1 for Pi > Pj) (batch_size, )
        - targets: Tensor of true labels for binary classification (batch_size, )

        Returns:
        - Combined loss value: LossBCE + LossRank
        """
        
        # LossBCE: Binary Cross-Entropy Loss
        loss_bce = self.bce_loss(Pi, targets)

        # LossRank: Ranking Loss
        loss_rank = torch.mean(torch.clamp(-y * (Pi - Pj) + self.margin, min=0))

        # Total Loss: Sum of BCE and Ranking Loss
        total_loss = loss_bce + loss_rank
        
        return total_loss


class Model(Encoder):
    """ task for entity linking"""
    def __init__(self, config, name=''):
        """ __init__ """
        ernie_config = config['ernie']
        if isinstance(ernie_config, str):
            ernie_config = json.loads(open(ernie_config).read())
            config['ernie'] = ernie_config
        super(Model, self).__init__(config, name=name)

        cls_config = config['cls_header']
        num_labels = cls_config['num_labels']

        linking_types = config.get('linking_types', {})
        self.start_cls = linking_types.get('start_cls', [])
        self.end_cls = linking_types.get('end_cls', [])
        assert len(self.start_cls) == len(self.end_cls)

        self.label_classifier = _build_linear(
                self.d_model,
                num_labels,
                append_name(name, 'labeling_cls'),
                nn.initializer.KaimingNormal())
        self.link_classifier = _build_linear(
                self.d_model,
                1,
                append_name(name, 'linking_cls'),
                nn.initializer.KaimingNormal())

    def kv_mask_gen(self, cls):
        mask = P.zeros([cls.shape[0], cls.shape[1], cls.shape[1]], 'int32')
        if len(self.start_cls) == 0:
            return P.ones_like(mask)
        for head, tail in zip(self.start_cls, self.end_cls):
            head_index = P.cast(cls == head, 'int32')
            tail_index = P.cast(cls == tail, 'int32')
            head_index = head_index.unsqueeze(2)
            tail_index = tail_index.unsqueeze(1).tile((1, tail_index.shape[1], 1))
            mask += head_index * tail_index

        mask = P.cast(mask > 0, 'int32')
        mask.stop_gradient = True
        return mask

    def forward(self, *args, **kwargs):
        """ forword """
        feed_names = kwargs.get('feed_names')
        input_data = dict(zip(feed_names, args))

        encoded, token_embeded = super(Model, self).forward(**input_data)
        encoded_2d = encoded.unsqueeze((1)) # [batch_size, 1, max_seqlen, d_model]

        batch_size = encoded.shape[0]
        max_seqlen = encoded.shape[1]

        link_label = input_data.get('link_label')
        seq_ids = input_data.get('sentence_ids')
        seq_mask = input_data.get('sentence_mask')
        cls_label = input_data.pop('cls_label')
        cls_mask = input_data.pop('label_mask') # [batch_size, line_num, line_num]
        link_mask = cls_mask.unsqueeze(-1).cast('float32') # [batch_size, line_num, 1]
        link_mask = link_mask.matmul(link_mask, transpose_y=True) # [batch_size, line_num, line_num]
        link_mask = link_mask * (1 - P.eye(link_mask.shape[1])).unsqueeze(0)
        link_mask = link_mask.cast('int32')

        max_line_num = cls_mask.shape[1]
        seq_ids = P.stack([(seq_ids == i).cast('float32') for i in range(1, max_line_num + 1)], axis=1) # [batch, line_num, max_seqlen]

        ## language token features
        lang_mask = P.cast(seq_mask == 0, 'int32') # [batch_size, max_seqlen]
        lang_ids = seq_ids * lang_mask.unsqueeze((1)) # [batch_size, line_num, max_seqlen]
        lang_ids = lang_ids.unsqueeze(-1)
        lang_emb = encoded_2d * lang_ids  # [batch_size, line_num, max_seqlen, d_model]
        lang_logit = lang_emb.sum(axis=2) / lang_ids.sum(axis=2).clip(min=1.0) # [batch_size, line_num, d_model]

        ## visual token features
        line_mask = P.cast(seq_mask == 1, 'int32') # [batch_size, max_seqlen]
        line_ids = seq_ids * line_mask.unsqueeze(1) # [batch_size, line_num, max_seqlen]
        line_ids = line_ids.unsqueeze(-1)
        line_emb = encoded_2d * line_ids
        line_logit = line_emb.sum(axis=2) / line_ids.sum(axis=2).clip(min=1.0) # [batch_size, line_num, d_model]

        ## later fusion
        encoder_emb = line_logit * lang_logit

        cls_logit = self.label_classifier(encoder_emb)
        cls_logit = P.argmax(cls_logit, axis=-1)

        mask = link_mask * self.kv_mask_gen(cls_logit)

        ## link loss calculation
        link_emb = encoder_emb.unsqueeze(1).tile((1, encoder_emb.shape[1], 1, 1))
        link_emb = P.abs(link_emb - link_emb.transpose((0, 2, 1, 3)))
        link_logit = self.link_classifier(link_emb).squeeze(-1)
        link_logit = F.sigmoid(link_logit) * mask

        return {'logit': link_logit, 'label': link_label}

    def eval(self):
        """ eval """
        if P.in_dynamic_mode():
            super(Model, self).eval()
        self.training = False
        for l in self.sublayers():
            l.training = False
        return self

    def train(self):
        """ train """
        if P.in_dynamic_mode():
            super(Model, self).train()
        self.training = True
        for l in self.sublayers():
            l.training = True
        return self

def train(model, data_loader, optimizer, num_epochs=10):
    model.train()
    criterion = CombinedLoss(margin=1.0)  # Use your CombinedLoss

    for epoch in range(num_epochs):
        for batch_data in data_loader:
            # Unpack batch data
            features, labels = batch_data
            input_data = {
                'feed_names': ['link_label', 'sentence_ids', 'sentence_mask', 'cls_label', 'label_mask'],
                'link_label': labels['link_label'],
                'sentence_ids': features['sentence_ids'],
                'sentence_mask': features['sentence_mask'],
                'cls_label': labels['cls_label'],
                'label_mask': labels['label_mask']
            }

            # Forward pass
            outputs = model(**input_data)
            link_logit = outputs['logit']
            link_label = outputs['label']

            # Compute loss
            targets = ...  # Define targets based on your data
            loss = criterion(link_logit, link_logit, link_label, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            print(f"Epoch {epoch}, Loss: {loss.numpy()}")

import paddle.optimizer as optim
config = {
    'ernie': 'path_to_ernie_config.json',
    'cls_header': {'num_labels': 10},
    'linking_types': {'start_cls': [1], 'end_cls': [2]}
}
model = Model(config)
optimizer = optim.Adam(parameters=model.parameters(), learning_rate=1e-4)
train(model, data_loader, optimizer)
