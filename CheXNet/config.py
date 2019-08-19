CUDA_NUM = 3
train_json = '../dataset/ChestX-ray8/small_train.json'
train_img_dir = '../dataset/ChestX-ray8/train_image/'
train_batch_size = 32

val_json = '../dataset/ChestX-ray8/small_val.json'
val_img_dir = '../dataset/ChestX-ray8/val_image/'
val_batch_size = 32

test_json = '../dataset/ChestX-ray8/small_test.json'
test_img_dir = '../dataset/ChestX-ray8/test_image/'
test_batch_size = 32

disease2id_json = '../dataset/ChestX-ray8/disease2id.json'

weights_dir = 'weights/'
epoch = 100
LR = 0.0001
factor = 0.1
patience = 1

