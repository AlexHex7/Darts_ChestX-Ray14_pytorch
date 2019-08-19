import os
import sys
import time
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import search_config as cfg
from lib.train_loader import TrainDataSet
from lib.val_loader import ValDataSet
from lib.metric import roc
from lib.architect import Architect
from lib.search_model import Network
import lib.utils as utils


SECOND_ORDER = False
if SECOND_ORDER:
    tag = 'second_order'
    cfg.unrolled = True
    cfg.CUDA_NUM = 7
else:
    tag = 'first_order'
    cfg.unrolled = False
    cfg.CUDA_NUM = 6
print(tag)

torch.cuda.set_device(cfg.CUDA_NUM)

# ------------------------- Log ------------------------------
log_dir = 'search_%s' % tag
os.makedirs(log_dir)

# --------------------------- Dataset ----------------------------
train_dataset = TrainDataSet(cfg)
val_dataset = ValDataSet(cfg)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# ------------------------ Model -----------------------------
criterion = nn.BCELoss()

model = Network(cfg.init_channels, cfg.CLS_NUM, cfg.layers, criterion)
model = model.cuda()

architect = Architect(model, cfg)

optimizer = torch.optim.SGD(model.parameters(), cfg.LR, cfg.momentum, cfg.weight_decay)
# optimizer = torch.optim.Adam(model.parameters(), cfg.LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs, eta_min=cfg.LR_min)
logging.info("param size = %fMB", utils.count_parameters_in_MB(model))


# ---------------------------------------------------------------
MAX_AUC = -1
for epoch_index in range(cfg.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]

    # ******************* training ************************
    torch.set_grad_enabled(True)
    model.train()

    total_loss = 0
    total_sample = 0
    for step, (img_batch, label_batch, _, _, _, _) in tqdm(enumerate(train_dataset.loader),
                                                           total=train_dataset.batches, ncols=0):
        img_batch = img_batch.cuda()
        label_batch = label_batch.cuda()

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(val_dataset.loader))

        input_search = input_search.cuda()
        target_search = target_search.cuda().float()

        architect.step(img_batch, label_batch, input_search, target_search, lr, optimizer, unrolled=cfg.unrolled)

        optimizer.zero_grad()
        pre_batch = model(img_batch)
        loss = criterion(pre_batch, label_batch.float())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += loss.item() * img_batch.size(0)
        total_sample += img_batch.size(0)

    mean_loss = total_loss / total_sample
    logging.info('epoch[%d/%d] LR[%.6f] train loss:%.4f' % (epoch_index, cfg.epochs, lr, mean_loss))

    # ************************ validation **********************
    torch.set_grad_enabled(False)
    model.eval()

    pre_list = []
    label_list = []
    for step, (img_batch, label_batch) in tqdm(enumerate(val_dataset.loader),
                                               total=val_dataset.batches, ncols=0):
        img_batch = img_batch.cuda()
        label_batch = label_batch.cuda()

        pre_batch = model(img_batch)
        loss = criterion(pre_batch, label_batch.float())

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

    if mean_auc > MAX_AUC:
        MAX_AUC = mean_auc

        with open(os.path.join(log_dir, 'genotype.txt'), 'w') as fp:
            fp.write('genotype = ' + str(model.genotype()))

    logging.info('valid_acc %.4f%% (%.4f%%)' % (mean_auc * 100, MAX_AUC * 100))

