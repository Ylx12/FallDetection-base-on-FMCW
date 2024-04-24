import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

def normalize_to_0_1(arr):
    min_val = torch.min(arr)
    max_val = torch.max(arr)
    normalized_arr = 1 * (arr - min_val) / (max_val - min_val)
    return normalized_arr

def make_dataset_list(folder_path, txt_dir, train_ratio):

    train_list, test_list = [], []

    for motion in os.listdir(folder_path):
        motion_dir = os.path.join(folder_path, motion)
        for top_path, sub_path, file in os.walk(motion_dir):
            if motion == 'fall':
                class_flag = 1
            else:
                class_flag = 0
            print(top_path)
            data_list = []
            for i in range(len(file)):
                data_list.append(os.path.join(top_path, file[i]))
            random.shuffle(data_list)

            for i in range(0, int(len(file) * train_ratio)):
                train_data = data_list[i] + ' ' + str(class_flag) + '\n'
                train_list.append(train_data)

            for i in range(int(len(file) * train_ratio), len(file)):
                test_data = data_list[i] + ' ' + str(class_flag) + '\n'
                test_list.append(test_data)

    print('train:', len(train_list), '   test:', len(test_list))

    random.shuffle(train_list)
    test_list.sort()
    train_list.sort()
    # random.shuffle(test_list)

    with open(txt_dir + 'train.txt', 'w', encoding='UTF-8') as f:
        for train_img in train_list:
            f.write(str(train_img))

    with open(txt_dir + 'test.txt', 'w', encoding='UTF-8') as f:
        for test_img in test_list:
            f.write(test_img)


class Fall_Dataset(Dataset):  # Generate two lists, one of which saves the data path and one of which saves the label corresponding to the data
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_label = self.load_annotations()  # Returns the dictionary. The file name is key and the label is value

        self.img = list(self.img_label.keys())
        self.label = [label for label in list(self.img_label.values())]

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = np.load(self.img[idx], allow_pickle=True)
        image = image.astype(np.float32)
        label = self.label[idx]
        label = torch.from_numpy(np.array(label))
        path = self.img[idx]

        return image, label, path

    def load_annotations(self):
        data_infos = {}
        with open(self.root_dir) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                data_infos[filename] = np.array(gt_label, dtype=np.int64)
        return data_infos


if __name__ == '__main__':

    radardata_path = '../data/'
    list_path = '../list/'

    if not os.path.exists(list_path):
        os.makedirs(list_path)
    for motion in os.listdir(radardata_path):
        motion_dir = os.path.join(radardata_path, motion)

    make_dataset_list(radardata_path, list_path, 0.7)


