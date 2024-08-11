import numpy as np

class STPMetric(object):
    def __init__(self, threshold, main_indicator='STP'):
        self.main_indicator = main_indicator
        self.threshold = threshold
        self.reset()

    def __call__(self, preds, labels, masks=None):
        if masks is None:
            masks = np.ones_like(labels)
        for pred_b, label_b, mask_b in zip(preds, labels, masks):
            for pred, label, mask in zip(pred_b, label_b, mask_b):
                if np.sum(mask.astype('int32')) == 0:
                    continue
                pos_p = np.array(pred[mask] > self.threshold, dtype='int32')
                pos_l = label[mask]
                
                # Check if the entire document is correctly predicted
                correct_document = np.all(pos_p == pos_l)
                if correct_document:
                    self.acc += 1

    def get_metric(self):
        res = {}
        res['STP'] = self.acc
        return res

    def reset(self):
        self.acc = 0
