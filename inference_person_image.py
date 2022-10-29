from config.config import PersonInferenceOptions
from models.person.AdaATModule import AdaATModule
from models.person.AdaATModule import kp2gaussian

import os
import torch
from collections import OrderedDict
import cv2
import numpy as np

if __name__ == '__main__':
    '''
    inference code of person image generation
    '''
    opt = PersonInferenceOptions().parse_args()
    model = AdaATModule(opt.img_channel, opt.keypoint_num).cuda()  #
    state_dict = torch.load(opt.inference_model_path)['state_dict']['net_g']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    ## source image
    source_img = cv2.imread(opt.source_img_path)
    source_tensor = torch.from_numpy(source_img / 255.0).permute(2, 0, 1).float().unsqueeze(0).cuda()
    ## source key points
    source_lm = np.loadtxt(opt.source_kp_path)
    source_lm[:, 0] = (source_lm[:, 0] / 176 * 2) - 1
    source_lm[:, 1] = (source_lm[:, 1] / 256 * 2) - 1
    source_lm_tensor = torch.from_numpy(source_lm).float().unsqueeze(0).cuda()
    ## visualize source heatmap
    source_lm_vis = kp2gaussian(source_lm_tensor, (256, 176), 0.01)
    source_lm_vis, _ = torch.max(source_lm_vis, dim=1, keepdim=True)
    source_lm_vis = source_lm_vis * 255
    source_lm_vis = source_lm_vis.cpu().squeeze().detach().numpy().astype(np.uint8)
    source_lm_vis = np.stack([source_lm_vis, source_lm_vis, source_lm_vis], 2)
    ## target key points
    target_lm = np.loadtxt(opt.target_kp_path)
    target_lm[:, 0] = (target_lm[:, 0] / 176 * 2) - 1
    target_lm[:, 1] = (target_lm[:, 1] / 256 * 2) - 1
    target_lm_tensor = torch.from_numpy(target_lm).float().unsqueeze(0).cuda()
    ## visualize target heatmap
    target_lm_vis = kp2gaussian(target_lm_tensor, (256, 176), 0.01)
    target_lm_vis,_ = torch.max(target_lm_vis, dim=1, keepdim=True)
    target_lm_vis = target_lm_vis * 255
    target_lm_vis = target_lm_vis.cpu().squeeze().detach().numpy().astype(np.uint8)
    target_lm_vis = np.stack([target_lm_vis, target_lm_vis, target_lm_vis], 2)
    with torch.no_grad():
        inference_out, _ = model(source_tensor, source_lm_tensor, target_lm_tensor)
        inference_out = inference_out * 255
        inference_out = inference_out.cpu().squeeze().permute(1, 2, 0).float().detach().numpy().astype(np.uint8)
        merge_img = np.concatenate([source_img, source_lm_vis, target_lm_vis, inference_out], 1)
        if os.path.exists(opt.res_person_path):
            os.remove(opt.res_person_path)
        cv2.imwrite(opt.res_person_path, merge_img)
        print('inference person image sucess!')


