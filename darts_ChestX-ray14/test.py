import os
import sys
import time
import torch
import logging
import torch.nn as nn

from lib.genotypes import Genotype
from lib.val_model import Network as Network
from lib.test_loader import TestDataSet
from lib.metric import roc
from tqdm import tqdm
import lib.utils as utils
import test_config as cfg


SECOND_ORDER = False
if SECOND_ORDER:
    tag = 'second_order'
    cfg.CUDA_NUM = 3
else:
    tag = 'first_order'
    cfg.CUDA_NUM = 2
print(tag)

genotype_path = os.path.join('search_%s' % tag, 'genotype.txt')
weight_path = os.path.join(cfg.weights_dir, '%s_net.pth' % tag)

print('genotype_path:', genotype_path)
print('weight_path:', weight_path)


torch.cuda.set_device(cfg.CUDA_NUM)
test_dataset = TestDataSet(cfg)


# ------------------------ Model -----------------------------
with open(genotype_path) as fp:
    line = 'genotype = ' + fp.readline()
exec(line)

model = Network(cfg.init_channels, cfg.CLS_NUM, cfg.layers, genotype)
state_dict = torch.load(weight_path)
model.load_state_dict(state_dict)
model = model.cuda()

print("param size = %fMB", utils.count_parameters_in_MB(model))

model.drop_path_prob = cfg.drop_path_prob

# -------------------- Test ----------------------
torch.set_grad_enabled(False)
model.eval()

pre_list = []
label_list = []
for step, (img_batch, label_batch) in tqdm(enumerate(test_dataset.loader),
                                           total=test_dataset.batches, ncols=0):
    img_batch = img_batch.cuda()
    label_batch = label_batch.cuda()

    pre_batch = model(img_batch)

    pre_list.append(pre_batch)
    label_list.append(label_batch)

pre_cat = torch.cat(pre_list, dim=0).cpu()
label_cat = torch.cat(label_list, dim=0).cpu()

auc_list = []
for disease_index in range(14):
    pre_slice = pre_cat[:, disease_index]
    label_slice = label_cat[:, disease_index]
    auc = roc(label_slice, pre_slice)
    auc_list.append(auc)

mean_auc = sum(auc_list) * 1.0 / auc_list.__len__()
record_line = 'test_acc %.4f%% (' + ('%.3f ' * 14) + ')'
print(record_line % (mean_auc * 100, *auc_list))

