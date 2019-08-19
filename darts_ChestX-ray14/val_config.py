train_json = '../dataset/ChestX-ray8/small_train.json'
train_img_dir = '../dataset/ChestX-ray8/train_image/'
train_batch_size = 32
train_shuffle = True

val_json = '../dataset/ChestX-ray8/small_val.json'
val_img_dir = '../dataset/ChestX-ray8/val_image/'
val_batch_size = 32
val_shuffle = False

disease2id_json = '../dataset/ChestX-ray8/disease2id.json'

weights_dir = 'weights/'

CUDA_NUM = 5
CLS_NUM = 14
LR = 0.025
# LR = 0.0001
momentum = 0.9
weight_decay = 3e-4
report_freq = 50
epochs = 500
init_channels = 36
layers = 20
cutout = False
cutout_length = 16
drop_path_prob = 0.2
grad_clip = 5

factor = 0.1
patience = 1
