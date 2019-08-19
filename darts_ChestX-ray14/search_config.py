train_json = '../dataset/ChestX-ray8/small_train.json'
train_img_dir = '../dataset/ChestX-ray8/train_image/'
train_batch_size = 16
train_shuffle = True

val_json = '../dataset/ChestX-ray8/small_val.json'
val_img_dir = '../dataset/ChestX-ray8/val_image/'
val_batch_size = 16
val_shuffle = True

disease2id_json = '../dataset/ChestX-ray8/disease2id.json'


CUDA_NUM = 6
CLS_NUM = 14
LR = 0.025
LR_min = 0.0001
arch_LR = 3e-4
arch_weight_decay = 1e-3
momentum = 0.9
weight_decay = 3e-4
epochs = 50

report_freq = 50

init_channels = 16
layers = 12

cutout = False
cutout_length = 16
drop_path_prob = 0.3
grad_clip = 5
unrolled = False



