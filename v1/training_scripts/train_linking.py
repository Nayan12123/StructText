""" entity linking at segment-level"""
import json
import paddle as P
import numpy as np
from paddle import nn
from model.encoder import Encoder
from paddle.nn import functional as F
import torch
from model.ernie.modeling_ernie import ACT_DICT, append_name, _build_linear, _build_ln

import torch
import torch.nn as Torchnn
import torch.nn.functional as TorchF
import os
import logging

class CombinedLoss(Torchnn.Module):
    def __init__(self, margin=1.0):
        super(CombinedLoss, self).__init__()
        self.margin = margin
        self.bce_loss = Torchnn.BCELoss()

    def forward(self, link_logit, link_label, mask):
        # Flatten the tensors
        link_logit_flat = link_logit.view(-1)
        link_label_flat = link_label.view(-1)
        mask_flat = mask.view(-1)

        # Apply mask
        valid_logits = link_logit_flat[mask_flat.bool()]
        valid_labels = link_label_flat[mask_flat.bool()]

        # Binary Cross-Entropy Loss
        loss_bce = self.bce_loss(valid_logits, valid_labels)

        # Margin Ranking Loss
        positive_samples = valid_logits[valid_labels.bool()]
        negative_samples = valid_logits[~valid_labels.bool()]

        # Ensure we have both positive and negative samples
        if len(positive_samples) > 0 and len(negative_samples) > 0:
            # Create all possible pairs
            P_i = positive_samples.unsqueeze(1).expand(-1, len(negative_samples))
            P_j = negative_samples.unsqueeze(0).expand(len(positive_samples), -1)

            # Compute margin ranking loss
            y = torch.ones_like(P_i)
            loss_rank = TorchF.margin_ranking_loss(P_i, P_j, y, margin=self.margin)
        else:
            loss_rank = torch.tensor(0.0, device=link_logit.device)

        # Combine losses
        total_loss = loss_bce + loss_rank

        return total_loss

class Model(Encoder):
    """Task for entity linking"""
    def __init__(self, config, name=''):
        """__init__"""
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

        # Freeze the base model parameters
        for param in self.parameters():
            param.stop_gradient = True

        # Ensure the classifiers are trainable
        for param in self.label_classifier.parameters():
            param.stop_gradient = False

        for param in self.link_classifier.parameters():
            param.stop_gradient = False

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
        """Forward"""
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
        """Eval"""
        if P.in_dynamic_mode():
            super(Model, self).eval()
        self.training = False
        for l in self.sublayers():
            l.training = False
        return self

    def train(self):
        """Train"""
        if P.in_dynamic_mode():
            super(Model, self).train()
        self.training = True
        for l in self.sublayers():
            # Set all layers to eval mode except for classifiers
            if l in [self.label_classifier, self.link_classifier]:
                l.training = True
            else:
                l.training = False
        return self

def train(model, data_loader, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    model.train()
    criterion = CombinedLoss(margin=1.0)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data in data_loader:
            # Move batch data to the correct device
            features = {k: v.to(device) for k, v in batch_data['features'].items()}
            labels = {k: v.to(device) for k, v in batch_data['labels'].items()}

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
            mask = labels['label_mask']
            loss = criterion(link_logit, link_label, mask)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    print("Training completed.")

def train(model, data_loader, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    model.train()
    criterion = CombinedLoss(margin=1.0)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data in data_loader:
            # Move batch data to the correct device
            features = {k: v.to(device) for k, v in batch_data['features'].items()}
            labels = {k: v.to(device) for k, v in batch_data['labels'].items()}

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
            mask = labels['label_mask']
            loss = criterion(link_logit, link_label, mask)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    print("Training completed.")




def resume_model(para_path, model):
        '''
        Resume from saved model
        :return:
        '''
        if os.path.exists(para_path):
            para_dict = P.load(para_path)
            model.set_dict(para_dict)
            logging.info('Load init model from %s', para_path)
            return model


model_config = {"architecture":{
        "ernie":"./configs/ernie_config/ernie_base.json",
        "cls_header":{
            "num_labels":4
        },
        "linking_types":{
          "start_cls":[1,2],
          "end_cls":[2,3]
        },
        "visual_backbone":{
            "module":"model.backbones.resnet_vd",
            "class":"ResNetVd",
            "params":{
                "layers":50
            },
            "fpn_dim":128
        },
        "embedding":{
            "roi_width":64,
            "roi_height":4,
            "rel_pos_size":36,
            "spa_pos_size":256,
            "max_seqlen":512,
            "max_2d_position_embedding":512
        }
    }}

model = Model(model_config, [
            "images",
            "sentence",
            "sentence_ids",
            "sentence_pos",
            "sentence_mask",
            "sentence_bboxes",
            "cls_label",
            "link_label",
            "label_mask"
        ])
base_model_path = ""
model = resume_model(base_model_path,model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = YourDataset(...)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model
model_config = {...}  # Your model configuration
model = Model(model_config)

# Load pre-trained weights
base_model_path = "path/to/your/pretrained/model.pdparams"
model = resume_model(base_model_path, model)

# Move model to the correct device
model.to(device)

# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Train the model
num_epochs = 10
train(model, train_loader, optimizer, num_epochs, device)

# Save the finetuned model
new_model_path = ""
P.save(model.state_dict(), new_model_path)