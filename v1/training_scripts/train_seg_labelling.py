""" sequence labeling at segment-level"""
import json
import paddle as P
import numpy as np
from model.encoder import Encoder
from paddle.nn import functional as F
import paddle.nn as nn
from paddle.optimizer import Adam
from model.ernie.modeling_ernie import ACT_DICT, append_name, _build_linear, _build_ln
import os
import logging
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 10

class Model(Encoder):
    """ task for entity labeling"""
    def __init__(self, config, name=''):
        """ __init__ """
        ernie_config = config['ernie']
        if isinstance(ernie_config, str):
            ernie_config = json.loads(open(ernie_config).read())
            config['ernie'] = ernie_config
        super(Model, self).__init__(config, name=name)

        cls_config = config['cls_header']
        num_labels = cls_config['num_labels']

        self.label_classifier = _build_linear(
                self.d_model,
                num_labels,
                append_name(name, 'labeling_cls'),
                nn.initializer.KaimingNormal())

    def forward(self, *args, **kwargs):
        """ forword """
        feed_names = kwargs.get('feed_names')
        input_data = dict(zip(feed_names, args))

        encoded, token_embeded = super(Model, self).forward(**input_data)
        encoded_2d = encoded.unsqueeze((1)) # [batch_size, 1, max_seqlen, d_model]

        seq_ids = input_data.get('sentence_ids')
        seq_mask = input_data.get('sentence_mask')
        label_mask = input_data.pop('label_mask') # [batch_size, line_num]

        max_line_num = label_mask.shape[1]
        seq_ids = P.stack([(seq_ids == i).cast('float32') for i in range(1, max_line_num + 1)], axis=1) # [batch, line_num, max_seqlen]

        lang_mask = P.cast(seq_mask == 0, 'int32') # [batch_size, max_seqlen]
        lang_ids = seq_ids * lang_mask.unsqueeze(1) # [batch_size, line_num, max_seqlen]
        lang_ids = lang_ids.unsqueeze(-1)
        lang_emb = encoded_2d * lang_ids  # [batch_size, line_num, max_seqlen, d_model]
        lang_logit = lang_emb.sum(axis=2) / lang_ids.sum(axis=2).clip(min=1.0) # [batch_size, line_num, d_model]

        line_mask = P.cast(seq_mask == 1, 'int32') # [batch_size, max_seqlen]
        line_ids = seq_ids * line_mask.unsqueeze(1) # [batch_size, line_num, max_seqlen]
        line_ids = line_ids.unsqueeze(-1)
        line_emb = encoded_2d * line_ids
        line_logit = line_emb.sum(axis=2) / line_ids.sum(axis=2).clip(min=1.0) # [batch_size, line_num, d_model]

        logit = line_logit * lang_logit
        logit = self.label_classifier(logit) # [batch_size, line_num, num_labels]

        label = input_data.get('label')
        logit = P.argmax(logit, axis=-1)
        mask = label_mask.cast('bool')

        selected_logit = P.masked_select(logit, mask)
        selected_label = P.masked_select(label, mask)
        mask = mask.cast('int32')

        return {'logit': selected_logit, 'label': selected_label,
                'logit_prim': logit, 'label_prim': label, 'mask': mask}

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
model = Model(model_config,[
            "images",
            "sentence",
            "sentence_ids",
            "sentence_pos",
            "sentence_mask",
            "sentence_bboxes",
            "label",
            "label_mask"
        ])
base_model_path = ""
model = resume_model(base_model_path,model)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(learning_rate=LEARNING_RATE, parameters=model.parameters())

# Assume we have a custom Dataset class
train_dataset = YourCustomDataset(...)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training loop
def train():
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch in train_loader:
            # Assuming batch contains all required inputs
            outputs = model(*batch)
            
            logits = outputs['logit']
            labels = outputs['label']
            
            loss = loss_fn(logits, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(train_loader)}")
    
    # Save the model after training
    P.save(model.state_dict(), 'trained_model.pdparams')
