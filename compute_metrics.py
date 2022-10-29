from config.config import MetricOptions

import torch
import lpips
import os
import glob
import numpy as np
import pytorch_ssim
import cv2


def addBounding(image, bound=40):
    ## when computing SSIM of person image, inspired from PATN, we padding image to 256x256
    ## the addBounding function is borrowed from PATN:https://github.com/tengteng95/Pose-Transfer/blob/master/tool/getMetrics_fashion.py
    h, w, c = image.shape
    image_bound = np.ones((h, w + bound * 2, c)) * 255
    image_bound = image_bound.astype(np.uint8)
    image_bound[:, bound:bound + w] = image
    return image_bound

if __name__ == '__main__':
    '''
    compute metrics of SSIM and LIPIS in deep fashion test data or voxceleb test data
    '''
    opt = MetricOptions().parse_args()
    inference_image_path_list = glob.glob(os.path.join(opt.inference_img_dir, '*.jpg'))
    loss_lpips = lpips.LPIPS().cuda()
    ssim_list = []
    lpips_list = []
    for img_index, inference_image_path in enumerate(inference_image_path_list):
        if opt.task_type == 'person':
            img_name = os.path.basename(inference_image_path).split('-')[1]
            real_image_path = os.path.join(opt.real_img_dir,img_name)
            if not os.path.exists(real_image_path):
                raise AssertionError
        else:
            img_name = os.path.basename(inference_image_path)
            real_image_path = os.path.join(opt.real_img_dir, img_name)
            if not os.path.exists(real_image_path):
                raise AssertionError
        ################################################ compute SSIM
        real_data = cv2.imread(real_image_path)
        if opt.task_type == 'person':
            real_data = addBounding(real_data)
        real_data = torch.from_numpy(np.rollaxis(real_data, 2)).float().unsqueeze(0)/255.0
        inference_data = cv2.imread(inference_image_path)
        if opt.task_type == 'person':
            inference_data = addBounding(inference_data)
        inference_data = torch.from_numpy(np.rollaxis(inference_data, 2)).float().unsqueeze(0)/255.0
        ssim_tem = pytorch_ssim.ssim(real_data, inference_data)
        ssim_list.append(ssim_tem)
        ################################################ compute LPIPS
        real_data_lpips = lpips.im2tensor(lpips.load_image(real_image_path)).cuda()
        inference_data_lpips = lpips.im2tensor(lpips.load_image(inference_image_path)).cuda()
        with torch.no_grad():
            lpips_tem = loss_lpips.forward(real_data_lpips, inference_data_lpips)
        lpips_list.append(float(lpips_tem))
        print('compute ssim&lpips {}/{} : name :{} ssim:{} lpips:{}'.format(img_index, len(inference_image_path_list),
                                                                    img_name,float(ssim_tem),float(lpips_tem)))
    mean_ssim = np.mean(np.array(ssim_list))
    mean_lpips = np.mean(np.array(lpips_list))
    print('the final metrics of {} images are ssim:{} lpips:{} on {} test data'.format(len(ssim_list),mean_ssim,mean_lpips,opt.task_type))


