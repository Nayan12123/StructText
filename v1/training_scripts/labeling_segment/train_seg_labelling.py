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
import dataset
import logging
import argparse
import functools
import json
from utils.build_dataloader import build_dataloader
import pickle
from utils.utility import add_arguments, print_arguments


LEARNING_RATE = 1e-3
NUM_EPOCHS = 3
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

class Model(Encoder):
    """ task for entity labeling"""
    def __init__(self, config, name=''):
        """ __init__ """
        ernie_config = config['ernie']
        if isinstance(ernie_config, str):
            ernie_config = json.loads(open(ernie_config).read())
            config['ernie'] = ernie_config
        super(Model, self).__init__(config, name=name)
        self.feed_names = name

        cls_config = config['cls_header']
        num_labels = cls_config['num_labels']

        self.label_classifier = _build_linear(
                self.d_model,
                num_labels,
                append_name(name, 'labeling_cls'),
                nn.initializer.KaimingNormal())

    def forward(self, *args, **kwargs):
        """ forword """
        # feed_names = kwargs.get('feed_names')
        print(len(args))
        print(args)
        input_data = dict(zip(self.feed_names, args))

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
        templogit = self.label_classifier(logit) # [batch_size, line_num, num_labels]

        label = input_data.get('label')
        print("logits before agrmax: ",templogit)
        logit = P.argmax(templogit, axis=-1)
        mask = label_mask.cast('bool')

        selected_logit = P.masked_select(logit, mask)
        selected_label = P.masked_select(label, mask)
        mask = mask.cast('int32')

        return {'logit': selected_logit, 'label': selected_label,'initial_logits':templogit,
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
            # para_dict = P.load(para_path)
            with open(para_path, "rb") as file:
                para_dict = pickle.load(file)
            model.set_dict(para_dict)
            logging.info('Load init model from %s', para_path)
            return model

# Training loop
loss_fn = nn.CrossEntropyLoss()

def train(model,train_loader):
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch in train_loader:
            # Assuming batch contains all required inputs
            outputs = model(*batch)
            
            logits = outputs['initial_logits']
            logits = logits.squeeze(0)
            print("logits======")
            print(logits)
            labels = outputs['label']
            print("labels")
            print(labels)
            # softmax = nn.Softmax(dim=0)
            # probabilities = softmax(logits)
            
            loss = loss_fn(logits, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(train_loader)}")
    
    # Save the model after training
    P.save(model.state_dict(), 'segment_labelling_trained_model.pdparams')

config = json.loads(open(args.config_file).read())
eval_config = config['eval']
model_config = config['architecture']
# model_config = {"architecture":{
#         "ernie":"./configs/ernie_config/ernie_base.json",
#         "cls_header":{
#             "num_labels":4
#         },
#         "linking_types":{
#           "start_cls":[1,2],
#           "end_cls":[2,3]
#         },
#         "visual_backbone":{
#             "module":"model.backbones.resnet_vd",
#             "class":"ResNetVd",
#             "params":{
#                 "layers":50
#             },
#             "fpn_dim":128
#         },
#         "embedding":{
#             "roi_width":64,
#             "roi_height":4,
#             "rel_pos_size":36,
#             "spa_pos_size":256,
#             "max_seqlen":512,
#             "max_2d_position_embedding":512
#         }
#     }}

# base_model_path = "./StrucTexT_base_pretrained.pdparams"
model = Model(model_config, eval_config['feed_names'])
print(model)
base_model_path = args.weights_path
print(base_model_path)
model = resume_model(base_model_path,model)
print(model)
optimizer = Adam(learning_rate=LEARNING_RATE, parameters=model.parameters())
config['init_model'] = base_model_path
# config['eval']['loader']['collect_batch'] = True
# config['eval']['loader']['batch_size_per_card']= args.batch_size
# config['eval']['loader']['shuffle'] = True

eval_config = config['eval']
eval_config['dataset']['data_path'] = args.label_path
eval_config['dataset']['image_path'] = args.image_path
eval_config['dataset']['max_seqlen'] = model_config['embedding']['max_seqlen']
# Assume we have a custom Dataset class
train_dataset = dataset.Dataset(
            eval_config['dataset'],
            eval_config['feed_names'],
            True)
place = P.set_device('cpu')

train_loader = build_dataloader(
    config['eval'],
    train_dataset,
    'Train',
    place,
    False)

train(model,train_loader)

##todo change batching size and hence update the config file's eval loader keys' batch size parameter
