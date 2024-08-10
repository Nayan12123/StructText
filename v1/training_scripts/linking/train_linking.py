""" entity linking at segment-level"""
import json
import paddle as P
import numpy as np
from paddle import nn
from model.encoder import Encoder
from paddle.nn import functional as F
from model.ernie.modeling_ernie import ACT_DICT, append_name, _build_linear, _build_ln
from paddle.optimizer import Adam
# import torch
# import torch.nn as Torchnn
# import torch.nn.functional as TorchF
import os
import logging
import dataset
import logging
import argparse
import functools
import json
from utils.build_dataloader import build_dataloader
import pickle
from utils.utility import add_arguments, print_arguments
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 10

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
parser = argparse.ArgumentParser('launch for training')
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--label_path', type=str, required=True)
parser.add_argument('--image_path', type=str, required=True)
parser.add_argument('--weights_path', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
args = parser.parse_args()
print_arguments(args)


import paddle
import paddle.nn.functional as F


class CombinedLoss(paddle.nn.Layer):
    def __init__(self, margin=1.0):
        super(CombinedLoss, self).__init__()
        self.margin = margin

    def forward(self, logits, true_labels):
        """
        :param logits: bs x m x m matrix of logits (bs = batch size, m = number of segments)
        :param true_labels: bs x m x m matrix of true binary labels
        :return: Combined Margin Ranking Loss and Binary Cross-Entropy Loss
        """
        # Sigmoid activation to convert logits to probabilities
        loss_bce = F.binary_cross_entropy(logits, true_labels.astype('float32'), reduction='mean')

        batch_size, m, _ = logits.shape
        positive_scores = logits * true_labels
        negative_mask = 1 - true_labels
        negative_scores = logits * negative_mask
        positive_scores_flat = positive_scores.reshape([batch_size, -1])
        negative_scores_flat = negative_scores.reshape([batch_size, -1])
        positive_scores_expanded = positive_scores_flat.unsqueeze(-1)  # Shape: [bs, m*m, 1]
        negative_scores_expanded = negative_scores_flat.unsqueeze(1)  # Shape: [bs, 1, m*m]
        y = paddle.ones_like(positive_scores_expanded)
        loss_ranking = F.margin_ranking_loss(
            positive_scores_expanded, 
            negative_scores_expanded, 
            y, 
            margin=self.margin
        )
        
        # Combine the BCE loss and Margin Ranking Loss
        loss = loss_bce + loss_ranking

        return loss



class Model(Encoder):
    """Task for entity linking"""
    def __init__(self, config, name=''):
        """__init__"""
        ernie_config = config['ernie']
        if isinstance(ernie_config, str):
            ernie_config = json.loads(open(ernie_config).read())
            config['ernie'] = ernie_config
        super(Model, self).__init__(config, name=name)
        self.feed_names = name

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
        # feed_names = kwargs.get('feed_names')
        
        input_data = dict(zip(self.feed_names, args))

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
        print(cls_logit.shape)
        mask = link_mask * self.kv_mask_gen(cls_logit)

        ## link loss calculation
        link_emb = encoder_emb.unsqueeze(1).tile((1, encoder_emb.shape[1], 1, 1))
        link_emb = P.abs(link_emb - link_emb.transpose((0, 2, 1, 3)))
        link_logit = self.link_classifier(link_emb).squeeze(-1)
        print(link_logit)
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
            # if l in [self.label_classifier, self.link_classifier]:
            #     l.training = True
            # else:
                l.training = False
        return self

# def train(model, data_loader, optimizer, num_epochs=10, device='cuda'):
#     model.to(device)
#     model.train()
#     criterion = CombinedLoss(margin=1.0)

#     for epoch in range(num_epochs):
#         total_loss = 0
#         for batch_data in data_loader:
#             # Move batch data to the correct device
#             features = {k: v.to(device) for k, v in batch_data['features'].items()}
#             labels = {k: v.to(device) for k, v in batch_data['labels'].items()}

#             input_data = {
#                 'feed_names': ['link_label', 'sentence_ids', 'sentence_mask', 'cls_label', 'label_mask'],
#                 'link_label': labels['link_label'],
#                 'sentence_ids': features['sentence_ids'],
#                 'sentence_mask': features['sentence_mask'],
#                 'cls_label': labels['cls_label'],
#                 'label_mask': labels['label_mask']
#             }

#             # Forward pass
#             outputs = model(**input_data)
#             link_logit = outputs['logit']
#             link_label = outputs['label']

#             # Compute loss
#             mask = labels['label_mask']
#             loss = criterion(link_logit, link_label, mask)

#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         avg_loss = total_loss / len(data_loader)
#         print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

#     print("Training completed.")

def train(model, data_loader):
    model.train()
    criterion = CombinedLoss(margin=1.0)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch_data in data_loader:
            # Move batch data to the correct device
            # print(batch_data.shape)  
                     
            outputs = model(*batch_data)

            # Forward pass
            link_logit = outputs['logit']
            link_label = outputs['label']
            print(link_label)
            

            # Compute loss
             
            # mask = batch_data[-1]        

            loss = criterion(link_logit, link_label)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")

    print("Training completed.")
    P.save(model.state_dict(), 'entity_linking_trained_model.pdparams')
    


def resume_model(para_path, model):
        '''
        Resume from saved model
        :return:
        '''
        if os.path.exists(para_path):
            # para_dict = P.load(para_path)
            with open(para_path, "rb") as file:
                para_dict = pickle.load(file)
            model.set_dict(para_dict)
            logging.info('Load init model from %s', para_path)
            return model


config = json.loads(open(args.config_file).read())
# config['eval']['loader']['collect_batch'] = True
# config['eval']['loader']['batch_size_per_card']= args.batch_size
eval_config = config['eval']
model_config = config['architecture']
model = Model(model_config, eval_config['feed_names'])
print(model)
base_model_path = args.weights_path
print(base_model_path)
model = resume_model(base_model_path,model)
print(model)
optimizer = Adam(learning_rate=LEARNING_RATE, parameters=model.parameters())
config['init_model'] = base_model_path
eval_config = config['eval']
eval_config['dataset']['data_path'] = args.label_path
eval_config['dataset']['image_path'] = args.image_path
eval_config['dataset']['max_seqlen'] = model_config['embedding']['max_seqlen']
# Assume we have a custom Dataset class
train_dataset = dataset.Dataset(
            eval_config['dataset'],
            eval_config['feed_names'],
            False)
place = P.set_device('cpu')

train_loader = build_dataloader(
    config['eval'],
    train_dataset,
    'Train',
    place,
    False)

train(model,train_loader)