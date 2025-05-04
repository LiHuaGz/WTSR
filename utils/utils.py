import os
import torch
from pytorch_msssim import ssim
import torch
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import cv2

def cal_psnr(HRs, SRs, max_value=255.0):
    # input: if channel is 3, then the input is YCrCb image; elif channel is 1, then the input is gray image
    # output: psnr
    with torch.no_grad():
        # 如果是YCrCb图像，转换为灰度图像
        if HRs.shape[1] == 3:
            HRs,SRs=HRs[:,0:1,:,:],SRs[:,0:1,:,:]
        mse = torch.mean((HRs - SRs) ** 2, dim=[1,2,3])  # 计算每个样本的MSE
        psnr = 10 * torch.log10(max_value ** 2 / mse)     # 计算每个样本的PSNR
        mpsnr = torch.mean(psnr)                          # 计算平均PSNR
    return mpsnr

def cal_ssim(HRs, SRs, max_value=255.0):
    # input: if channel is 3, then the input is YCrCb image; elif channel is 1, then the input is gray image
    # output: ssim
    with torch.no_grad():
        # 如果是YCrCb图像，转换为灰度图像
        if HRs.shape[1] == 3:
            HRs,SRs=HRs[:,0:1,:,:],SRs[:,0:1,:,:]
        ssim_value = ssim(HRs, SRs, data_range=max_value, size_average=True)
    return ssim_value

def set_seed(seed=0):
    # Python 和 NumPy 的随机种子
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch 的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU

    # 确保 CuDNN 结果可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_checkpoint(model, optimizer, scheduler, scaler, opt, filename='checkpoint.pt'):
    start_epoch = 0
    best_psnr = 0.0
    result = None
    if os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        try:
            # Map storage location based on availability
            if torch.cuda.is_available():
                map_location = lambda storage, loc: storage.cuda()
            else:
                map_location = 'cpu'

            ckpt = torch.load(filename, map_location=map_location, weights_only=False) # weights_only=False to load optimizer etc.
            if 'epoch' in ckpt and 'epochs' in ckpt and ckpt['epoch'] == ckpt['epochs']:
                start_epoch = 0  # Reset to 0 if we've completed all epochs
                epochs = opt.epochs
                print(f"Training completed in checkpoint (epoch {ckpt['epoch']} of {ckpt['epochs']}). Restarting from epoch 0.")
            elif 'epoch' in ckpt and 'epochs' in ckpt and ckpt['epoch'] != ckpt['epochs']:
                start_epoch = ckpt['epoch']
                epochs = ckpt['epochs']
                print(f"Didn't complete trainning in checkpoint (epoch {ckpt['epoch']} of {ckpt['epochs']}). Restarting to train.")
            else:
                start_epoch = 0
                epochs = opt.epochs
                print(f"Resuming training from epoch {start_epoch}")
            # opt = ckpt.get('opt', opt) # Careful about overwriting current opt
            best_psnr = ckpt.get('best_psnr', 0.0) # Load best_psnr
            result = ckpt.get('result', None) # Load result list

            # Load model state
            try:
                model.load_state_dict(ckpt['model_state_dict'])
            except RuntimeError as e:
                 print(f"Warning: Could not load model state dict strictly: {e}. Trying without strict loading.")
                 model.load_state_dict(ckpt['model_state_dict'], strict=False)


            # Load optimizer state
            if 'optimizer_state_dict' in ckpt and optimizer is not None:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as e:
                    print(f"Warning: Could not load optimizer state dict: {e}")
            else:
                 print("Optimizer state not found in checkpoint or optimizer is None.")


            # Load scheduler state
            if 'scheduler_state_dict' in ckpt and scheduler is not None:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except Exception as e:
                    print(f"Warning: Could not load scheduler state dict: {e}")
            else:
                 print("Scheduler state not found in checkpoint or scheduler is None.")


            # Load scaler state
            if 'scaler_state_dict' in ckpt and scaler is not None:
                try:
                    scaler.load_state_dict(ckpt['scaler_state_dict']) # Load scaler state
                except Exception as e:
                    print(f"Warning: Could not load scaler state dict: {e}")
            else:
                 print("Scaler state not found in checkpoint or scaler is None.")


            # Restore RNG states if available
            if 'rng_state' in ckpt:
                rng_state = ckpt['rng_state']
                try:
                    random.setstate(rng_state['python_random'])
                    np.random.set_state(rng_state['numpy_random'])
                    torch.set_rng_state(rng_state['torch_cpu'])
                    if torch.cuda.is_available() and rng_state.get('torch_cuda') is not None:
                        torch.cuda.set_rng_state(rng_state['torch_cuda'])
                    print(f"Restored RNG state from epoch {start_epoch}")
                except Exception as e:
                    print(f"Warning: Could not restore RNG state: {e}. Resetting seed.")
                    set_seed(opt.seed) # Fallback
            else:
                print("RNG state not found in checkpoint, setting initial seed.")
                set_seed(opt.seed) # Fallback

            print(f"=> loaded checkpoint '{filename}' (epoch {start_epoch})")

        except Exception as e:
            print(f"Error loading checkpoint '{filename}': {e}")
            print("Starting from scratch.")
            start_epoch = 0 # Reset epoch if loading fails
            epochs=opt.epochs
            best_psnr = 0.0
            result = None
            set_seed(opt.seed) # Ensure seed is set if loading fails

    else:
        print(f"=> no checkpoint found at '{filename}', starting from scratch")
        epochs=opt.epochs
        set_seed(opt.seed) # Ensure seed is set if starting fresh

    return start_epoch, epochs, opt, best_psnr, result # Return loaded/default values

def cv2_lanczos_interpolate(tensor, scale_factor):
    b, c, h, w = tensor.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    result = torch.zeros((b, c, new_h, new_w), device=tensor.device)
    
    def process_channel(args):
        i, j, img, new_h, new_w, device = args
        img_np = img.cpu().numpy()
        resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)  # OpenCV Lanczos interpolation
        return i, j, torch.from_numpy(resized).to(device)
    
    args_list = [(i, j, tensor[i, j], new_h, new_w, tensor.device) 
                for i in range(b) for j in range(c)]
    
    # 并行计算
    with ThreadPoolExecutor() as executor:
        for i, j, resized_tensor in executor.map(process_channel, args_list):
            result[i, j] = resized_tensor
    return result