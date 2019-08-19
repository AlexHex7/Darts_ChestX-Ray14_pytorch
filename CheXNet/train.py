import torch
from torch import nn
from torch import optim
import config as cfg
from lib.train_loader import TrainDataSet
from lib.val_loader import ValDataSet
from tqdm import tqdm
from lib.metric import roc, f1_score_calc
import os
from lib.network import Network

torch.cuda.set_device(cfg.CUDA_NUM)

print('Train_Dir', cfg.train_img_dir)
print('Train_Json', cfg.train_json)
print('Train_Batch_Size', cfg.train_batch_size)
print('Val_Dir', cfg.val_img_dir)
print('Val_Json', cfg.val_json)
print('Val_Batch_Size', cfg.val_batch_size)
print('')
print('LR', cfg.LR)
print('Decay Factor', cfg.factor)
print('Patience', cfg.patience)


train_set = TrainDataSet(cfg)
val_set = ValDataSet(cfg)

net = Network()
net.cuda(cfg.CUDA_NUM)

opt = optim.Adam(net.parameters(), lr=cfg.LR)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
lr_scheduler = lr_scheduler(opt, factor=cfg.factor, mode='max', patience=cfg.patience)

bce_func = nn.BCELoss()

MAX_AUC = 0
STEP = 0
for epoch_index in range(cfg.epoch):
    # ------------------- Training ------------------------
    torch.set_grad_enabled(True)
    net.train(True)
    for img, label, disease_num, view, gender, age \
            in tqdm(train_set.loader, total=train_set.batches, disable=False, ncols=0):
        bs = img.size(0)
        img = img.cuda(cfg.CUDA_NUM)
        label = label.cuda(cfg.CUDA_NUM).float()

        prediction = net(img)

        loss = bce_func(prediction, label)
        opt.zero_grad()
        loss.backward()
        opt.step()

        STEP += 1

    print('\t(LR:%.6f)[%d/%d] Step:%d loss:%.4f [MAX_AUC: %.3f]'
          % (opt.param_groups[0]['lr'], epoch_index + 1, cfg.epoch,
             STEP, loss.item(), MAX_AUC)
          )

    # ------------------- Validating ------------------------
    torch.set_grad_enabled(False)
    net.eval()

    pre_list = []
    label_list = []
    for val_img, val_label \
            in tqdm(val_set.loader, total=val_set.batches, disable=False, ncols=0):
        val_img = val_img.cuda(cfg.CUDA_NUM)
        val_label = val_label.cuda(cfg.CUDA_NUM)
        b, c, h, w = val_img.size()

        prediction = net(val_img)

        pre_list.append(prediction)
        label_list.append(val_label)

    pre_cat = torch.cat(pre_list, dim=0).cpu()
    label_cat = torch.cat(label_list, dim=0).cpu()

    auc_list = []
    for disease_index in range(14):
        pre_slice = pre_cat[:, disease_index]
        label_slice = label_cat[:, disease_index]

        auc = roc(label_slice, pre_slice)
        auc_list.append(auc)

    mean_auc = sum(auc_list) * 1.0 / auc_list.__len__()

    lr_scheduler.step(mean_auc)

    details_mode = '(%s)' % ('%.3f ' * 14)
    AUC_PRINT = '\t[max_auc:%.3f] [mean_auc:%.3f] ' + details_mode

    AUC_SHOW = AUC_PRINT % (MAX_AUC, mean_auc, *auc_list)
    print(AUC_SHOW)

    with open('log', 'a+') as fp:
        fp.write(AUC_SHOW + '\n')

    if mean_auc > MAX_AUC:
        MAX_AUC = mean_auc
        weights_path = os.path.join(cfg.weights_dir, 'net.pth')
        torch.save(net.state_dict(), weights_path)
