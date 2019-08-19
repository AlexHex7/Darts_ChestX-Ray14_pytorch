test_json = '../dataset/ChestX-ray8/small_test.json'
test_img_dir = '../dataset/ChestX-ray8/test_image/'
test_batch_size = 32
test_shuffle = False

disease2id_json = '../dataset/ChestX-ray8/disease2id.json'

weights_dir = 'weights/'
genotype_dir = 'genotype/'

CUDA_NUM = 1
CLS_NUM = 14
init_channels = 36
layers = 20

cutout = False
cutout_length = 16

drop_path_prob = 0.2
grad_clip = 5

auxiliary = False
auxiliary_weight = 0.4
