from torch.utils.data import DataLoader
import argparse
import tqdm
import torch
import torch.nn as nn
import os
import cv2
import numpy as np

# import self-defined functions
from dataloaders import MyImageDataset
from model import SRnet_dtcwt
from utils.utils import cal_psnr, cal_ssim, set_seed, cv2_lanczos_interpolate

def interference(opt, saveimg=True):
    # set seed
    set_seed(3407)

    # set device
    device = torch.device(opt.device)

    # define model: dtcwt
    model_dtcwt = SRnet_dtcwt(biort=opt.biort, qshift=opt.qshift).to(device)
    ckpt_dtcwt = torch.load(opt.weight, weights_only=False, map_location=device)
    model_dtcwt.load_state_dict(ckpt_dtcwt['model_state_dict'])
    
    # 检查当前路径下是否有opt.save_dir对应文件夹，没有则创建
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    # 检查opt.save_dir文件夹下是否有epoch，没有则创建，有则检查其后缀数字，取最大值加1，作为当前epoch文件夹
    if not os.path.exists(os.path.join(opt.save_dir, 'result')):
        save_dir = os.path.join(opt.save_dir, 'result')
        os.makedirs(save_dir)
    else:
        save_dir = os.path.join(opt.save_dir, 'result'+str(len(os.listdir(opt.save_dir))-1))
        os.makedirs(save_dir)
    img_dir=os.path.join(save_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    # load data
    test_dir=os.path.join(opt.dataset, 'test')
    test_data = MyImageDataset(image_folder=test_dir, transform=None, isTrain=False,
                                opt=opt
                                )
    test_loader = DataLoader(test_data, 
                             batch_size=opt.batch_size, 
                             shuffle=False, 
                             pin_memory=True,
                             num_workers=opt.num_workers)
    
    # eval
    model_dtcwt.eval()

    running_psnr_dtcwtJ2, running_psnr_bicubic, running_psnr_lanczos = 0.0, 0.0, 0.0
    running_ssim_dtcwtJ2, running_ssim_bicubic, running_ssim_lanczos = 0.0, 0.0, 0.0
    pbar = tqdm.tqdm(test_loader, desc='final test & saving images', total=len(test_loader))
    with torch.no_grad():
        for batch_index,(LR_Y, LR_Cr, LR_Cb, HR_Y, HR_Cr, HR_Cb, imgs_name) in enumerate(pbar):
            # 把数据放到GPU上
            LR_Y, HR_Y = LR_Y.to(device), HR_Y.to(device)
            LR_Cr, LR_Cb = LR_Cr.to(device), LR_Cb.to(device)
            HR_Cr, HR_Cb = HR_Cr.to(device), HR_Cb.to(device)

            # 正向传播
            SR_Y_dtcwtJ2 = model_dtcwt(LR_Y)

            # 计算LR
            LR = torch.cat((LR_Y, LR_Cr, LR_Cb), dim=1)

            # 计算HR
            HR = torch.cat((HR_Y, HR_Cr, HR_Cb), dim=1)

            # 计算SR
            SR_Cr = nn.functional.interpolate(LR_Cr, scale_factor=2, mode='bicubic', align_corners=False)
            SR_Cb = nn.functional.interpolate(LR_Cb, scale_factor=2, mode='bicubic', align_corners=False)
            SR_dtcwtJ2 = torch.cat((SR_Y_dtcwtJ2, SR_Cr, SR_Cb), dim=1)
            
            # 计算SR_bicubic
            SR_Y_bicubic = nn.functional.interpolate(LR_Y, scale_factor=2, mode='bicubic', align_corners=False)
            SR_bicubic = torch.cat((SR_Y_bicubic, SR_Cr, SR_Cb), dim=1)

            # 计算SR_Lanczos
            SR_Y_Lanczos = cv2_lanczos_interpolate(LR_Y, scale_factor=2)
            SR_Lanczos = torch.cat((SR_Y_Lanczos, SR_Cr, SR_Cb), dim=1)

            # 计算psnr
            running_psnr_dtcwtJ2 += cal_psnr(HR_Y, SR_Y_dtcwtJ2, max_value=255.0)
            running_psnr_bicubic += cal_psnr(HR_Y, SR_Y_bicubic,max_value=255.0)
            running_psnr_lanczos += cal_psnr(HR_Y, SR_Y_Lanczos, max_value=255.0)

            # 计算SSIM
            running_ssim_dtcwtJ2 += cal_ssim(HR_Y, SR_Y_dtcwtJ2, max_value=255.0)
            running_ssim_bicubic += cal_ssim(HR_Y, SR_Y_bicubic, max_value=255.0)
            running_ssim_lanczos += cal_ssim(HR_Y, SR_Y_Lanczos, max_value=255.0)

            # 实时更新显示psnr
            mpsnr_dtcwtJ2, mssim_dtcwtJ2 = running_psnr_dtcwtJ2/(batch_index+1), running_ssim_dtcwtJ2/(batch_index+1)
            mpsnr_bicubic, mssim_bicubic = running_psnr_bicubic/(batch_index+1), running_ssim_bicubic/(batch_index+1)
            mpsnr_lanczos, mssim_lanczos = running_psnr_lanczos/(batch_index+1), running_ssim_lanczos/(batch_index+1)

            # 使用progress bar的refresh参数和set_postfix
            """ pbar.set_postfix({
                'PSNR_dtcwtJ2': f"{mpsnr_dtcwtJ2:.4f}",
                'SSIM_dtcwtJ2': f"{mssim_dtcwtJ2:.4f}",
                'PSNR_bicubic': f"{mpsnr_bicubic:.4f}",
                'SSIM_bicubic': f"{mssim_bicubic:.4f}",
                'PSNR_Lanczos': f"{mpsnr_lanczos:.4f}",
                'SSIM_Lanczos': f"{mssim_lanczos:.4f}"
            }) """

            # save images
            if saveimg:
                for i in range(HR_Y.shape[0]):
                    img_name=imgs_name[i].split('.')[0]  # 去掉文件后缀
                    
                    # 保存SR_dtcwtJ2图片
                    img_SR_dtcwtJ2 = SR_dtcwtJ2[i].cpu().numpy().transpose(1,2,0)
                    img_SR_dtcwtJ2 = np.clip(img_SR_dtcwtJ2, 0, 255).astype(np.uint8)
                    img_SR_dtcwtJ2 = cv2.cvtColor(img_SR_dtcwtJ2, cv2.COLOR_YCrCb2BGR)
                    cv2.imwrite(os.path.join(img_dir, img_name + '_SR_dtcwt.png'), img_SR_dtcwtJ2)
                    
                    # 保存SR_bicubic图片
                    img_bicubic = SR_bicubic[i].cpu().numpy().transpose(1,2,0)
                    img_bicubic = np.clip(img_bicubic, 0, 255).astype(np.uint8)
                    img_bicubic = cv2.cvtColor(img_bicubic, cv2.COLOR_YCrCb2BGR)
                    cv2.imwrite(os.path.join(img_dir, img_name + '_bicubic.png'), img_bicubic)

                    # 保存SR_Lanczos图片
                    img_Lanczos = SR_Lanczos[i].cpu().numpy().transpose(1,2,0)
                    img_Lanczos = np.clip(img_Lanczos, 0, 255).astype(np.uint8)
                    img_Lanczos = cv2.cvtColor(img_Lanczos, cv2.COLOR_YCrCb2BGR)
                    cv2.imwrite(os.path.join(img_dir, img_name + '_Lanczos.png'), img_Lanczos)

                    # 保存HR图片
                    img_HR = HR[i].cpu().numpy().transpose(1,2,0)
                    img_HR = np.clip(img_HR, 0, 255).astype(np.uint8)
                    img_HR = cv2.cvtColor(img_HR, cv2.COLOR_YCrCb2BGR)
                    cv2.imwrite(os.path.join(img_dir, img_name + '_HR.png'), img_HR)

                    # 保存LR图片
                    img_LR = LR[i].cpu().numpy().transpose(1,2,0)
                    img_LR = np.clip(img_LR, 0, 255).astype(np.uint8)
                    img_LR = cv2.cvtColor(img_LR, cv2.COLOR_YCrCb2BGR)
                    cv2.imwrite(os.path.join(img_dir, img_name + '_LR.png'), img_LR)

    print(f"Average PSNR_dtcwtJ2: {mpsnr_dtcwtJ2:.4f}, Average SSIM_dtcwtJ2: {mssim_dtcwtJ2:.4f}")
    print(f"Average PSNR_bicubic: {mpsnr_bicubic:.4f}, Average SSIM_bicubic: {mssim_bicubic:.4f}")
    print(f"Average PSNR_Lanczos: {mpsnr_lanczos:.4f}, Average SSIM_Lanczos: {mssim_lanczos:.4f}")

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='test')
    parser.add_argument('--weight', type=str, default='weight/dtcwt.pt')
    parser.add_argument('--biort', type=str, default='near_sym_b')
    parser.add_argument('--qshift', type=str, default='qshift_b')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pre_load', type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--block_size', type=int, default=64)
    parser.add_argument('--scale_factor', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early_stop', type=int, default=30)
    return parser.parse_known_args()[0] if known else parser.parse_args()

if __name__ == '__main__':
    # 将相对目录转为当前文件所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 解析命令行参数
    opt=parse_opt()
    opt.pre_load = (opt.pre_load == 'True')

    #开始训练
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    interference(opt)