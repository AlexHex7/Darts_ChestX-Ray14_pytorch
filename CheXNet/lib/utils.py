import config as cfg
import numpy as np


def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        if param_group['lr'] < cfg.MIN_LR:
            param_group['lr'] = cfg.MIN_LR

            
def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

