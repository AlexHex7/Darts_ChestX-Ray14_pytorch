import os
import sys
import time
import torch
import logging
import torch.nn as nn

from lib.genotypes import Genotype
from lib.val_model import Network
from lib.train_loader import TrainDataSet
from lib.val_loader import ValDataSet
from lib.metric import roc
from tqdm import tqdm
import lib.utils as utils
import val_config as cfg

SECOND_ORDER = True
if SECOND_ORDER:
    tag = 'second_order'
    cfg.CUDA_NUM = 5
else:
    tag = 'first_order'
    cfg.CUDA_NUM = 4
print(tag)

genotype_path = os.path.join('search_%s' % tag, 'genotype.txt')
log_path = 'log_val_%s.txt' % tag
weight_path = os.path.join(cfg.weights_dir, '%s_net.pth' % tag)

print('genotype_path:', genotype_path)
print('log_path:', log_path)
print('weight_path:', weight_path)

torch.cuda.set_device(cfg.CUDA_NUM)

# ------------------------- Log ------------------------------
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(log_path)
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# --------------------------- Dataset ----------------------------
train_dataset = TrainDataSet(cfg)
val_dataset = ValDataSet(cfg)

# ------------------------ Model -----------------------------
with open(genotype_path) as fp:
    line = fp.readline()

exec(line)
model = Network(cfg.init_channels, cfg.CLS_NUM, cfg.layers, genotype)
model = model.cuda()
logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

criterion = nn.BCELoss()
criterion = criterion.cuda()

optimizer = torch.optim.SGD(model.parameters(), cfg.LR, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
# optimizer = torch.optim.Adam(model.parameters(), cfg.LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
# scheduler = lr_scheduler(optimizer, factor=cfg.factor, mode='max', patience=cfg.patience)


MAX_AUC = -1
for epoch_index in range(cfg.epochs):
    scheduler.step()
    model.drop_path_prob = cfg.drop_path_prob * epoch_index / cfg.epochs

    # --------------------- Train ------------------
    torch.set_grad_enabled(True)
    model.train()

    total_loss = 0
    total_sample = 0
    for step, (img_batch, label_batch, _, _, _, _) in tqdm(enumerate(train_dataset.loader),
                                                           total=train_dataset.batches,
                                                           ncols=0):

        img_batch = img_batch.cuda()
        label_batch = label_batch.cuda()

        optimizer.zero_grad()
        pre_batch = model(img_batch)
        loss = criterion(pre_batch, label_batch.float())
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += loss.item() * img_batch.size(0)
        total_sample += img_batch.size(0)

    mean_loss = total_loss / total_sample
    logging.info('epoch[%d/%d] LR[%.6f] train loss:%.4f' % (epoch_index, cfg.epochs,
                                                            # optimizer.param_groups[0]['lr'],
                                                            scheduler.get_lr()[0],
                                                            mean_loss))

    # ------------------------ Valid -------------------------------
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

    # scheduler.step(mean_auc)

    if mean_auc > MAX_AUC:
        MAX_AUC = mean_auc

        model.cpu()
        torch.save(model.state_dict(), weight_path)
        model.cuda()

    record_line = 'valid_acc %.4f%% (%.4f%%) (' + ('%.3f ' * 14) + ')'
    logging.info(record_line % (mean_auc * 100, MAX_AUC * 100, *auc_list))
