from torch.utils.data import DataLoader
import argparse
import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import os
import cv2
from pytorch_wavelets import DTCWTForward
from torch.amp import autocast, GradScaler  # 从 torch.amp 导入而不是 torch.cuda.amp
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# import self-defined functions
from dataloaders import MyImageDataset
from model import SRnet_dtcwt
from utils.utils import cal_psnr, set_seed, load_checkpoint

def computeloss(HR, SR, low, pred_low, high1, pred_high1, high2, pred_high2):
    return 0.8*torch.mean((HR-SR)**2) + 0.2*(torch.mean((low-pred_low)**2) + torch.mean((high1-pred_high1)**2) + torch.mean((high2-pred_high2)**2))

def train(opt):
    # set start time
    start_time = time.time()

    # 初始化随机种子
    set_seed(opt.seed)

    # set device
    device = torch.device(opt.device)

    # 定义小波变换类
    wt=DTCWTForward(J=2, biort=opt.biort, qshift=opt.qshift, include_scale=[True, False]).to(device)

    # 定义model
    model = SRnet_dtcwt(biort=opt.biort, qshift=opt.qshift, device=opt.device).to(device)

    # auto
    scaler = GradScaler()
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    # load checkpoint, weight_file是当前文件夹下的weight文件夹下的文件名
    weight = opt.weight
    start_epoch, epochs, opt, best_psnr, result = load_checkpoint(model, optimizer, None, scaler, opt, filename=weight)

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
    last=os.path.join(save_dir, 'last.pt')
    best=os.path.join(save_dir, 'best.pt')
    txt_dir=os.path.join(save_dir, 'result.txt')
    img_dir=os.path.join(save_dir, 'images')

    # load data
    train_dir=os.path.join(opt.dataset, 'train')
    test_dir=os.path.join(opt.dataset, 'test')
    train_data = MyImageDataset(image_folder=train_dir, transform=None, isTrain=True, 
                                opt=opt
                                )
    valid_data = MyImageDataset(image_folder=test_dir, transform=None, isTrain=False,
                                opt=opt
                                )
    test_data = MyImageDataset(image_folder=test_dir, transform=None, isTrain=False,
                                opt=opt
                                )
    train_loader = DataLoader(train_data, 
                              batch_size=opt.batch_size,
                              shuffle=True, 
                              pin_memory=True, 
                              num_workers=opt.num_workers,
                              persistent_workers=True, 
                              )
    valid_loader = DataLoader(valid_data, 
                             batch_size=1, 
                             shuffle=False, 
                             pin_memory=True,
                             num_workers=opt.num_workers,
                             persistent_workers=True, ) 
    test_loader = DataLoader(test_data, 
                             batch_size=1, 
                             shuffle=False, 
                             pin_memory=True,
                             num_workers=opt.num_workers) 
    
    # scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        total_steps=len(train_loader) * opt.epochs,
        pct_start=0.03,  # 前3%步数用于预热
        div_factor=100,   # 初始学习率=1e-4/100=1e-6
        final_div_factor=100,  # 最终学习率=1e-8
    )

    # 初始化结果
    if result is None: # 如果 checkpoint 中没有 result
        result = []
    if best_psnr is None: # 如果 checkpoint 中没有 best_psnr
        best_psnr = 0.0
    
    # train
    #--------------------training--------------------#
    model.train()
    for epoch in range(start_epoch,epochs):
        # start epoch time
        epoch_start_time = time.time()

        running_loss = 0.0
        running_psnr = 0.0
        pbar = tqdm.tqdm(train_loader, desc='training', total=len(train_loader))
        #-------------------start batch-------------------#
        for i,(LR_Y, _, _, HR_Y, _, _,  _) in enumerate(pbar):
            # 清空梯度
            optimizer.zero_grad()
            
            # 把数据放到GPU上
            LR_Y, HR_Y = LR_Y.to(device), HR_Y.to(device)

            # 使用混合精度进行前向传播
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu',enabled=False):
                SR_Y = model(LR_Y.float())

                # HR小波系数
                batch_size, _, height, width = HR_Y.shape
                low, high = wt(HR_Y.float())
                low1, high1, high2 = low[0], high[0], high[1]
                high1 = high1.permute(0,3,4,1,2,5).reshape(batch_size, height//2, width//2, 12).permute(0,3,1,2)
                high2 = high2.permute(0,3,4,1,2,5).reshape(batch_size, height//4, width//4, 12).permute(0,3,1,2)

                # SR小波系数
                pred_low, pred_high = wt(SR_Y.float())
                pred_low1, pred_high1, pred_high2 = pred_low[0], pred_high[0], pred_high[1]
                pred_high1 = pred_high1.permute(0,3,4,1,2,5).reshape(batch_size, height//2, width//2, 12).permute(0,3,1,2)
                pred_high2 = pred_high2.permute(0,3,4,1,2,5).reshape(batch_size, height//4, width//4, 12).permute(0,3,1,2)

                #计算损失
                loss = computeloss(HR_Y.float(), SR_Y.float(), low1, pred_low1, high1, pred_high1, high2, pred_high2)

            # 使用scaler进行反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 计算损失
            running_loss += loss.item()

            # calculate psnr
            running_psnr += cal_psnr(HR_Y, SR_Y, max_value=255.0)

            # 实时更新显示loss和psnr
            mloss_train, mpsnr_train = running_loss/(i+1), running_psnr/(i+1)
            pbar.set_description(f'Epoch [{epoch+1}/{epochs}]')
            pbar.set_postfix(Loss=f'{mloss_train:.4f}', PSNR=f'{mpsnr_train:.4f}')

            # 每隔几个批次清理一次缓存
            if i % 10 == 0:
                torch.cuda.empty_cache()
        #-------------------end batch-------------------#
        # scheduler
        scheduler.step()

        if (epoch+1) % 1 == 0:
            mloss_valid,mpsnr_valid=test(model, valid_loader, saveimg=False, img_dir=img_dir, opt=opt)

        # end epoch time
            epoch_end_time = time.time()

        # save results and time cost
            with open(txt_dir, 'a') as f:
                f.write(f"Epoch [{epoch+1}/{epochs}], Loss_train: {mloss_train:.4f}, PSNR_train: {mpsnr_train:.4f}, Loss_valid: {mloss_valid:.4f}, PSNR_valid: {mpsnr_valid:.4f}, Time_cost: {epoch_end_time - epoch_start_time:.2f} seconds\n")
            result.append([mloss_train, mpsnr_train, mloss_valid, mpsnr_valid])

        # save checkpoint
            rng_state = {
                'python_random': random.getstate(),
                'numpy_random': np.random.get_state(),
                'torch_cpu': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            }

            ckpt={
                'epoch': epoch+1,
                'epochs': epochs,
                'opt': opt,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(), 
                'rng_state': rng_state, 
                'best_psnr': best_psnr, 
                'result': result 
            }

        # 检查是否NaN
            if torch.isnan(mloss_valid.clone().detach()) or torch.isnan(mpsnr_valid.clone().detach()) or mpsnr_valid == float('inf') or mpsnr_valid == float('-inf') or mloss_valid == float('inf') or mloss_valid == float('-inf'):
                print('Invalid loss or PSNR (NaN or inf detected), skip saving checkpoint and stop training.')
                epoch = epochs
                break
            else:
                torch.save(ckpt, last)
                if mpsnr_valid > best_psnr:
                    print(f"Epoch {epoch+1}: New best PSNR: {mpsnr_valid:.4f} > {best_psnr:.4f}. Saving best checkpoint.")
                    best_psnr = mpsnr_valid 
                    torch.save(ckpt, best)
                else:
                     print(f"Epoch {epoch+1}: PSNR: {mpsnr_valid:.4f} did not improve from {best_psnr:.4f}.")

        if (epoch+1) % 5 == 0:
            # 画出结果
            with torch.no_grad():
                result_tensor = torch.tensor(result)
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.plot(result_tensor[:, 0], label='train loss')
                plt.plot(result_tensor[:, 2], label='valid loss')
                plt.legend()
                plt.subplot(1, 2, 2)
                plt.plot(result_tensor[:, 1], label='train PSNR')
                plt.plot(result_tensor[:, 3], label='valid PSNR')
                plt.legend()
                plt.savefig(os.path.join(save_dir, 'result.png'))
                plt.close()
            
    # 载入best.pt并测试
    ckpt = torch.load(best, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
    # end time
    end_time = time.time()

    # test
    mloss, mpsnr = test(model, test_loader, saveimg=True, img_dir=img_dir, opt=opt)
    print(f'Finished Training, best loss: {mloss:.4f}, best PSNR: {mpsnr:.4f}, results saved at {save_dir}')
    print(f'Training time: {end_time - start_time:.2f} seconds')

    # save time cost
    with open(txt_dir, 'a') as f:
        f.write(f"Training time: {end_time - start_time:.2f} seconds\n")
    #--------------------end train--------------------#

def test(model, test_loader, saveimg=False, img_dir='images', opt=None):
    device = torch.device(opt.device)

    # 定义小波变换类
    wt=DTCWTForward(J=2, biort=opt.biort, qshift=opt.qshift, include_scale=[True, False]).to(device)

    model.eval()

    running_loss = 0.0
    running_psnr = 0.0

    if not saveimg:
        pbar = tqdm.tqdm(test_loader, desc='testing', total=len(test_loader))
    else:
        pbar = tqdm.tqdm(test_loader, desc='final test & saving images', total=len(test_loader))
    with torch.no_grad():
        for batch_index,(LR_Y, LR_Cr, LR_Cb, HR_Y, _, _, imgs_name) in enumerate(pbar):
            # 把数据放到GPU上
            LR_Y, HR_Y = LR_Y.to(device), HR_Y.to(device)
            LR_Cr, LR_Cb = LR_Cr.to(device), LR_Cb.to(device)
            
            # 前向传播
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=False):
                SR_Y = model(LR_Y.float())
                
                # HR小波系数
                batch_size, _, height, width = HR_Y.shape
                low, high = wt(HR_Y.float())
                low1, high1, high2 = low[0], high[0], high[1]
                high1 = high1.permute(0,3,4,1,2,5).reshape(batch_size, height//2, width//2, 12).permute(0,3,1,2)
                high2 = high2.permute(0,3,4,1,2,5).reshape(batch_size, height//4, width//4, 12).permute(0,3,1,2)

                # SR小波系数
                pred_low, pred_high = wt(SR_Y.float())
                pred_low1, pred_high1, pred_high2 = pred_low[0], pred_high[0], pred_high[1]
                pred_high1 = pred_high1.permute(0,3,4,1,2,5).reshape(batch_size, height//2, width//2, 12).permute(0,3,1,2)
                pred_high2 = pred_high2.permute(0,3,4,1,2,5).reshape(batch_size, height//4, width//4, 12).permute(0,3,1,2)

                #计算损失
                loss = computeloss(HR_Y.float(), SR_Y.float(), low1, pred_low1, high1, pred_high1, high2, pred_high2)

            # 计算损失和PSNR
            running_loss += loss
            running_psnr += cal_psnr(HR_Y, SR_Y, max_value=255.0)
            
            # 实时更新显示loss和psnr
            mloss, mpsnr = running_loss/(batch_index+1), running_psnr/(batch_index+1)
            pbar.set_postfix(Loss=f'{mloss:.4f}', PSNR=f'{mpsnr:.4f}')

            # save images
            if saveimg:
                os.makedirs(img_dir, exist_ok=True)

                if isinstance(SR_Y, list):
                    SR_Y = torch.cat(SR_Y, dim=0)
                
                for i in range(SR_Y.shape[0]):
                    # 处理色彩通道, 对色度通道进行插值到相同尺寸
                    _, _, sr_h, sr_w = SR_Y[i:i+1].shape
                    SR_Cr = nn.functional.interpolate(LR_Cr[i:i+1], size=(sr_h, sr_w), 
                                                     mode='bicubic', align_corners=False)
                    SR_Cb = nn.functional.interpolate(LR_Cb[i:i+1], size=(sr_h, sr_w),
                                                     mode='bicubic', align_corners=False)
                    
                    # 合并通道
                    SR = torch.cat((SR_Y[i:i+1], SR_Cr, SR_Cb), dim=1).detach().cpu().numpy()
                    
                    # 保存图像
                    img = SR[0]  # 第一个维度是batch
                    img = np.transpose(img, (1, 2, 0)) 
                    img = np.clip(img, 0, 255).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
                    cv2.imwrite(os.path.join(img_dir, imgs_name[i]), img)

    return mloss, mpsnr
    
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='train')
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

def main(opt):
    # 将相对目录转为当前文件所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 解析命令行参数
    opt.pre_load = (opt.pre_load == 'True')

    #开始训练
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train(opt)

if __name__ == '__main__':
    opt=parse_opt()
    main(opt)