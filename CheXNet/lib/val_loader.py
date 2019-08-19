import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms
import json
import os


class ValDataSetDefine(data.Dataset):
    def __init__(self, item_list, d2id, config):
        self.item_list = item_list
        self.d2id = d2id
        self.cfg = config

        # to_tensor = transforms.ToTensor()
        # tensor_normal = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.transform = transforms.Compose([
        #     transforms.TenCrop(size=[448, 448]),
        #     transforms.Lambda(lambda crops: torch.stack([to_tensor(crop) for crop in crops])),
        #     transforms.Lambda(lambda crops: torch.stack([tensor_normal(crop) for crop in crops])),
        # ])

        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        """
        info_dict: 'label', 'follow-up', 'pid', 'age', 'gender', 'view'
        :param index:
        :return: (1, 1, h, w), (1, 1, h, w)
        """

        img_name, info_dict = self.item_list[index]

        # --------------- Read Image ------------------
        img_path = os.path.join(self.cfg.val_img_dir, img_name)
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = self.transform(img)

        # --------------- Constuct class ID ------------------
        label = [0] * 14
        disease = info_dict['label']

        if disease != 'No Finding':
            d_list = disease.split('|')
            for a_disease in d_list:
                index = self.d2id[a_disease]
                label[index] = 1

            assert d_list.__len__() == sum(label)

        label = torch.LongTensor(label)
        return img, label

    def __len__(self):
        return self.item_list.__len__()


class ValDataSet(object):
    def __init__(self, config):
        self.cfg = config

        # ---------- Get Dict[disease] to id ------------------
        with open(config.disease2id_json) as fp:
            self.d2id = json.load(fp)

        # --------- Get img_list and label_list ---------------
        with open(config.val_json) as fp:
            json_dict = json.load(fp)

        # for img_name in json_dict.keys():
        #     img_dict = json_dict[img_name]
        #     label = img_dict['label']
        #     view_point = img_dict['view']

        print('\n================== From val_loader.py ==================')
        self.item_list = list(json_dict.items())
        self.samples = self.item_list.__len__()

        self.dataset = ValDataSetDefine(self.item_list, self.d2id, config)
        self.loader = data.DataLoader(dataset=self.dataset, batch_size=self.cfg.val_batch_size,
                                      shuffle=False, num_workers=10)
        self.batches = self.loader.__len__()

        print('Validating:(%d batches) (%d samples)' % (self.batches, self.samples))
        print('===========================================================\n')


if __name__ == '__main__':
    import sys
    sys.path.append('../')

    import config as cfg

    cfg.test_json = '../../dataset/ChestX-ray8/test.json'
    cfg.test_img_dir = '../../dataset/ChestX-ray8/test_image/'
    cfg.disease2id_json = '../../dataset/ChestX-ray8/disease2id.json'
    cfg.test_batch_size = 16

    obj = TestDataSet(cfg)

    # i = 0
    for a, b in obj.loader:
        print(a.size(), b.size())
        break

    #     i += 1
    #     print(img0.size(), label0.size())
        # pass