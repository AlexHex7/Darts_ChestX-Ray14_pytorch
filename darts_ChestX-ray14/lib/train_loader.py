import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms
import json
import os


class TrainDataSetDefine(data.Dataset):
    def __init__(self, item_list, d2id, config):
        self.item_list = item_list
        self.d2id = d2id
        self.cfg = config

        self.transform = transforms.Compose([
            # transforms.Resize([224, 224]),
            # transforms.RandomResizedCrop(size=[224, 224]),
            transforms.RandomCrop([224, 224]),
            transforms.RandomHorizontalFlip(0.5),
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
        img_path = os.path.join(self.cfg.train_img_dir, img_name)
        with Image.open(img_path) as img:
            # print(img_name, img)
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

        # ------------------ Disease num ----------------------
        dn_label = torch.LongTensor([0])
        dn_label[0] = sum(label)

        # ------------------- Gender ---------------------
        # Male [0]; Female [1];
        g_label = torch.LongTensor([0])
        gender = info_dict['gender']
        assert gender in ['M', 'F']
        if gender == 'F':
            g_label[0] = 1

        # -------------------- View ----------------------
        # PA [0]; AP [1];
        v_label = torch.LongTensor([0])
        view = info_dict['view']
        assert view in ['PA', 'AP']
        if view == 'AP':
            v_label[0] = 1

        # --------------------- Age Line ------------------
        # 1~10 [0], 11~20 [1], ...91~100+ [9]
        a_label = torch.LongTensor([0])
        age = int(info_dict['age'])
        for index, divide in enumerate(range(10, 101, 10)):
            if divide == 100:
                divide = 10000

            if age <= divide:
                a_label[0] = index
                break

        return img, label, dn_label, v_label, g_label, a_label,

    def __len__(self):
        return self.item_list.__len__()


class TrainDataSet(object):
    def __init__(self, config):
        self.cfg = config

        # ---------- Get Dict[disease] to id ------------------
        with open(config.disease2id_json) as fp:
            self.d2id = json.load(fp)

        # --------- Get img_list and label_list ---------------
        with open(config.train_json) as fp:
            json_dict = json.load(fp)

        # for img_name in json_dict.keys():
        #     img_dict = json_dict[img_name]
        #     label = img_dict['label']
        #     view_point = img_dict['view']

        print('\n================== From train_loader.py ==================')
        self.item_list = list(json_dict.items())
        self.samples = self.item_list.__len__()

        self.dataset = TrainDataSetDefine(self.item_list, self.d2id, config)
        self.loader = data.DataLoader(dataset=self.dataset, batch_size=self.cfg.train_batch_size,
                                      shuffle=config.train_shuffle, num_workers=10)
        self.batches = self.loader.__len__()

        print('Training:(%d batches) (%d samples)' % (self.batches, self.samples))
        print('===========================================================\n')


if __name__ == '__main__':
    import sys
    sys.path.append('../')

    import config as cfg

    cfg.train_json = '../dataset/ChestX-ray8/train.json'
    cfg.train_img_dir = '../dataset/ChestX-ray8/train_image/'
    cfg.disease2id_json = '../dataset/ChestX-ray8/disease2id.json'
    cfg.train_batch_size = 16

    obj = TrainDataSet(cfg)

    # i = 0
    for a, b, c, d, e, f in obj.loader:
        print(a.size(), b.size(), c.size(), d.size(), e.size(), f.size())
        print(b[0])
        print(c[0])
        print(d[0])
        print(e[0])
        print(f[0])
        break

    #     i += 1
    #     print(img0.size(), label0.size())
        # pass