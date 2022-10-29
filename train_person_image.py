from models.common.Discriminator import Discriminator 
from models.common.VGG19 import Vgg19
from models.person.AdaATModule import AdaATModule
from utils import get_scheduler, update_learning_rate
from config.config import PersonTrainingOptions
from utils import GANLoss
from dataset.dataset_person import Data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import os
import torch.nn.functional as F

if __name__ == "__main__":
    '''
    training code of person image generation
    '''
    # load config
    opt = PersonTrainingOptions().parse_args()
    # set seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # load training data in memory
    train_data = Data(opt.train_data,opt.train_img_dir)
    training_data_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    train_data_length = len(training_data_loader)
    # init network
    net_g = AdaATModule(opt.img_channel, opt.keypoint_num).cuda()
    net_d = Discriminator(opt.img_channel + opt.keypoint_num, opt.D_block_expansion, opt.D_num_blocks,
                           opt.D_max_features).cuda()
    net_vgg = Vgg19().cuda()
    # parallel
    net_g = nn.DataParallel(net_g)
    net_d = nn.DataParallel(net_d)
    net_vgg = nn.DataParallel(net_vgg)
    # set optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr_g)
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr_d)
    # resume
    if opt.resume != 'None':
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        opt.start_epoch = checkpoint['epoch']
        net_g_static = checkpoint['state_dict']['net_g']
        net_g.load_state_dict(net_g_static)
        net_d.load_state_dict(checkpoint['state_dict']['net_d'])
        optimizer_g.load_state_dict(checkpoint['optimizer']['net_g'])
        optimizer_d.load_state_dict(checkpoint['optimizer']['net_d'])
    # set criterion
    criterionGAN = GANLoss().cuda()
    criterionL1 = nn.L1Loss().cuda()
    # set scheduler
    net_g_scheduler = get_scheduler(optimizer_g, opt.non_decay, opt.decay)
    net_d_scheduler = get_scheduler(optimizer_d, opt.non_decay, opt.decay)

    # start train
    for epoch in range(opt.start_epoch, opt.non_decay + opt.decay + 1):
        net_g.train()
        for iteration, data in enumerate(training_data_loader):
            # read data
            source_tensor, source_fitting_lm,target_tensor,target_fitting_lm = data
            source_tensor = source_tensor.float().cuda()
            source_fitting_lm = source_fitting_lm.float().cuda()
            target_tensor = target_tensor.float().cuda()
            target_fitting_lm = target_fitting_lm.float().cuda()
            # network forward
            fake_out, target_heatmap = net_g(source_tensor,source_fitting_lm,target_fitting_lm)
            # down sample output image and real image
            fake_out_half = F.avg_pool2d(fake_out, 3, 2, 1, count_include_pad=False)
            target_tensor_half = F.interpolate(target_tensor, scale_factor=0.5, mode='bilinear')
            # (1) Update D network
            optimizer_d.zero_grad()
            # compute fake loss
            condition_fake_d = torch.cat([fake_out, target_heatmap], 1)
            _,pred_fake_d = net_d(condition_fake_d)
            loss_d_fake = criterionGAN(pred_fake_d, False)
            # compute real loss
            condition_real_d = torch.cat([target_tensor, target_heatmap], 1)
            _,pred_real_d = net_d(condition_real_d)
            loss_d_real = criterionGAN(pred_real_d, True)
            # Combine D loss
            loss_dI = (loss_d_fake + loss_d_real) * 0.5
            loss_dI.backward(retain_graph=True)
            optimizer_d.step()
            # (2) Update G network
            _, pred_fake_dI = net_d(condition_fake_d)
            optimizer_g.zero_grad()
            # compute perception loss
            perception_real = net_vgg(target_tensor)
            perception_fake = net_vgg(fake_out)
            perception_real_half = net_vgg(target_tensor_half)
            perception_fake_half = net_vgg(fake_out_half)
            loss_g_perception = 0
            for i in range(len(perception_real)):
                loss_g_perception += criterionL1(perception_fake[i], perception_real[i])
                loss_g_perception += criterionL1(perception_fake_half[i], perception_real_half[i])
            loss_g_perception = (loss_g_perception / len(perception_real) ) * opt.lamb_perception
            # gan dI loss
            loss_g_dI = criterionGAN(pred_fake_dI, True)
            # combine perception loss and gan loss
            loss_g = loss_g_perception + loss_g_dI
            loss_g.backward()
            optimizer_g.step()

            print(
                "===> Epoch[{}]({}/{}): Loss_DI: {:.4f} Loss_GI: {:.4f} Loss_perception: {:.4f} lr_g = {:.7f} lr_d = {:.7f}".format(
                    epoch, iteration, len(training_data_loader), float(loss_dI), float(loss_g_dI),
                    float(loss_g_perception), optimizer_g.param_groups[0]['lr'], optimizer_d.param_groups[0]['lr']))

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)

        # checkpoint
        if epoch % opt.checkpoint == 0:
            if not os.path.exists(opt.result_path):
                os.mkdir(opt.result_path)
            model_out_path = os.path.join(opt.result_path, 'person_epoch_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'state_dict': {'net_g': net_g.state_dict(), 'net_d': net_d.state_dict()},
                'optimizer': {'net_g': optimizer_g.state_dict(), 'net_d': optimizer_d.state_dict()}
            }
            torch.save(states, model_out_path)
            print("Checkpoint saved to {}".format(epoch))


