import torch
from torch import nn
import config as cfg
from lib.test_loader import TestDataSet
from tqdm import tqdm
from lib.metric import roc, f1_score_calc
import os
from lib.network import Network
import time
from lib.utils import count_parameters_in_MB

torch.cuda.set_device(cfg.CUDA_NUM)

test_set = TestDataSet(cfg)

net = Network()

print('Net contain %.2fMB' % count_parameters_in_MB(net))
net.cuda()

weight_path = os.path.join(cfg.weights_dir, 'net.pth')
net.load_state_dict(torch.load(weight_path))

# ------------------- Testing ------------------------
torch.set_grad_enabled(False)
net.eval()

pre_list = []
label_list = []

for test_img, test_label \
        in tqdm(test_set.loader, total=test_set.batches, disable=False, ncols=0):
    test_img = test_img.cuda()
    test_label = test_label.cuda()

    b, c, h, w = test_img.size()
    prediction = net(test_img)

    pre_list.append(prediction)
    label_list.append(test_label)

pre_cat = torch.cat(pre_list, dim=0)
label_cat = torch.cat(label_list, dim=0)

auc_list = []
f1_list = []
for disease_index in range(14):
    pre_slice = pre_cat[:, disease_index].cpu()
    label_slice = label_cat[:, disease_index].cpu()

    auc = roc(label_slice, pre_slice)

    auc_list.append(auc)

mean_auc = sum(auc_list) * 1.0 / auc_list.__len__()
line = 'MEAN_AUC:%.3f\n' + '%.3f ' * 14

print(line % (mean_auc, *auc_list))

