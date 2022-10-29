import cv2
import json
import os
import numpy as np
from torch.utils.data import Dataset
import torch

def get_data(train_json_path,train_img_dir):
    print('loading data....')
    with open(train_json_path, 'r') as f:
        data_list = json.load(f)
    data_length = len(data_list)
    data_dic = {}
    for frame_pair_index,frame_pair in enumerate(data_list):
        print('loading {}/{} data'.format(frame_pair_index,data_length))
        source_img_name = frame_pair['source_image']
        if source_img_name not in data_dic:
            data_dic[source_img_name] = cv2.imread(os.path.join(train_img_dir,source_img_name))
        target_img_name = frame_pair['target_image']
        if target_img_name not in data_dic:
            data_dic[target_img_name] = cv2.imread(os.path.join(train_img_dir,target_img_name))
    return data_list,data_dic


class Data(Dataset):
    def __init__(self,train_json_path,train_img_dir):
        super(Data, self).__init__()
        self.data_list,self.img_dic= get_data(train_json_path,train_img_dir)
        self.length = len(self.data_list)


    def __getitem__(self, index):
        frame_pair = self.data_list[index]
        ## source img
        source_data = self.img_dic[frame_pair['source_image']]
        source_data = source_data/255.0
        ## source fitting lm
        source_fitting_lm = np.array(frame_pair['source_lm'])
        source_fitting_lm[:,0] = (source_fitting_lm[:,0] / 176 * 2) - 1
        source_fitting_lm[:,1] = (source_fitting_lm[:, 1] / 256 * 2) - 1

        ## target img
        target_data = self.img_dic[frame_pair['target_image']]
        target_data = target_data / 255.0
        ## target fitting lm
        target_fitting_lm = np.array(frame_pair['target_lm'])
        target_fitting_lm[:,0] = (target_fitting_lm[:,0] / 176 * 2) - 1
        target_fitting_lm[:, 1] = (target_fitting_lm[:, 1] / 256 * 2) - 1

        # tensor
        source_tensor = torch.from_numpy(source_data).float().permute(2, 0, 1)
        source_fitting_lm = torch.from_numpy(source_fitting_lm).float()
        target_tensor = torch.from_numpy(target_data).float().permute(2, 0, 1)
        target_fitting_lm = torch.from_numpy(target_fitting_lm).float()
        return source_tensor, source_fitting_lm,target_tensor,target_fitting_lm

    def __len__(self):
        return self.length



