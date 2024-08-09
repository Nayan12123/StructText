""" evaler.py """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import logging
import numpy as np
import paddle as P
from tqdm import trange

class Evaler:
    """
    Evaler class
    """

    def __init__(self, config, model, data_loader, eval_classes=None):
        '''
        :param config:
        :param model:
        :param data_loader:
        '''
        self.model = model
        self.eval_classes = eval_classes
        self.valid_data_loader = data_loader
        self.len_step = len(self.valid_data_loader)

        self.init_model = config['init_model']
        self.valid_config = config['eval']

    @P.no_grad()
    def run(self):
        '''
        print evaluation results
        '''
        self._resume_model()
        self.model.eval()
        for eval_class in self.eval_classes.values():
            eval_class.reset()

        total_time = 0.0
        total_frame = 0.0
        t = trange(self.len_step)
        loader = self.valid_data_loader()
        for step_idx in t:
            t.set_description('evaluate with example %i' % step_idx)
            input_data = next(loader)
            start = time.time()
            feed_names = self.valid_config['feed_names']
            output = self.model(*input_data, feed_names=feed_names)
            total_time += time.time() - start

            ####### Eval ##########
            for key, val in self.eval_classes.items():
                if 'entity' in key and 'label_prim' in output.keys():
                    label = output['label_prim'].numpy()
                    logit = output['logit_prim'].numpy()
                    mask = output.get('mask', None)
                else:
                    label = output['label'].numpy()
                    logit = output['logit'].numpy()
                    mask = output.get('mask', None)
                mask = None if mask is None else mask.numpy()
                val(logit, label, mask)
            #########################
            total_frame += input_data[0].shape[0]
        metrics = 'fps : {}'.format(total_frame / total_time)
        for key, val in self.eval_classes.items():
            metrics += '\n{}:\n'.format(key) + str(val.get_metric())
        print('[Eval Validation] {}'.format(metrics))

    def _resume_model(self):
        '''
        Resume from saved model
        :return:
        '''
        para_path = self.init_model
        if os.path.exists(para_path):
            para_dict = P.load(para_path)
            self.model.set_dict(para_dict)
            logging.info('Load init model from %s', para_path)



""" trainer.py """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import logging
import numpy as np
import paddle as P
from tqdm import trange
from paddle.optimizer import Adam
from paddle.nn import CrossEntropyLoss

class Trainer:
    """
    Trainer class for training a model
    """

    def __init__(self, config, model, data_loader):
        '''
        :param config: Configuration dictionary
        :param model: Model instance
        :param data_loader: DataLoader instance
        '''
        self.model = model
        self.data_loader = data_loader
        self.len_step = len(self.data_loader)
        self.config = config

        # Initialize optimizer and loss function
        self.optimizer = Adam(learning_rate=config['learning_rate'], parameters=self.model.parameters())
        self.loss_fn = CrossEntropyLoss()

        # Path for saving the model
        self.save_path = config['save_path']
        self.model_state = config.get('init_model', None)

    def train(self):
        '''
        Training procedure
        '''
        if self.model_state:
            self._resume_model()

        self.model.train()

        total_time = 0.0
        total_loss = 0.0
        t = trange(self.len_step)

        loader = self.data_loader()
        for step_idx in t:
            t.set_description('Training step %i' % step_idx)
            input_data = next(loader)
            inputs, labels = input_data  # Assume input_data is a tuple (inputs, labels)

            start = time.time()
            
            # Forward pass
            outputs = self.model(*inputs)
            logits = outputs['logit']
            loss = self.loss_fn(logits, labels)

            # Backward pass and optimization
            self.optimizer.clear_grad()
            loss.backward()
            self.optimizer.step()

            total_time += time.time() - start
            total_loss += loss.numpy().item()

            # Print progress
            t.set_postfix(loss=total_loss / (step_idx + 1))

            # Optionally save model periodically
            if (step_idx + 1) % self.config.get('save_interval', 1000) == 0:
                self._save_model()

        print('[Training Completed] Average Loss: {}'.format(total_loss / self.len_step))

    def _resume_model(self):
        '''
        Resume from saved model
        '''
        para_path = self.model_state
        if os.path.exists(para_path):
            para_dict = P.load(para_path)
            self.model.set_dict(para_dict)
            logging.info('Loaded model from %s', para_path)

    def _save_model(self):
        '''
        Save model state
        '''
        P.save(self.model.state_dict(), self.save_path)
        logging.info('Model saved to %s', self.save_path)

