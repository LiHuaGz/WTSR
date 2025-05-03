import os
import cv2
import random
import torch
from torch.utils.data import Dataset
import pywt
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import psutil

class MyImageDataset(Dataset):
    def __init__(self, image_folder, transform=None, isTrain=False,
                  opt=None):
        self.scale_factor = opt.scale_factor
        self.HRs_folder = os.path.join(image_folder, 'HR')
        self.HRs_paths = [os.path.join(self.HRs_folder, f) for f in os.listdir(self.HRs_folder)
                            if f.lower().endswith(('.jpg', '.png'))]
        self.LRs_folder = os.path.join(image_folder, 'LR')
        self.isTrain = isTrain
        self.preload = opt.pre_load
        num_images = len(self.HRs_paths)
        self.HRs = [None] * num_images
        self.img_names = [None] * num_images
        self.num_workers = opt.num_workers
        self.block_size = opt.block_size
        self.transform = transform

        # 使用多线程预加载图像
        if self.preload:
            print(f"预加载{len(self.HRs_paths)}张图像到内存...")

            def load_HRs(args):
                idx, img_path = args
                # 检查文件是否存在且可读
                if os.path.exists(img_path) and os.access(img_path, os.R_OK):
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"警告: 无法读取图像文件 {img_path}")
                            return idx, None, None # 返回 None 表示加载失败
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                        return idx, img, os.path.basename(img_path)
                    except Exception as e:
                        print(f"错误: 加载图像 {img_path} 时出错: {e}")
                        return idx, None, None # 返回 None 表示加载失败
                else:
                    print(f"警告: 图像文件不存在或不可读 {img_path}")
                    return idx, None, None # 返回 None 表示加载失败

            valid_indices = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(tqdm(
                    executor.map(load_HRs, enumerate(self.HRs_paths)),
                    total=len(self.HRs_paths)
                ))

            # 过滤掉加载失败的图像
            original_paths = self.HRs_paths
            self.HRs_paths = []
            temp_HRs = []
            temp_img_names = []
            for idx, img, img_name in results:
                if img is not None and img_name is not None:
                    # 只有成功加载的才保留
                    temp_HRs.append(img)
                    temp_img_names.append(img_name)
                    self.HRs_paths.append(original_paths[idx]) # 保留对应的有效路径

            # 更新列表
            self.HRs = temp_HRs
            self.img_names = temp_img_names

            if not self.HRs:
                 raise ValueError("错误：未能成功加载任何 HR 图像。请检查 HR 文件夹和图像文件。")

            print(f"图像预加载完成！成功加载 {len(self.HRs)} 张图像。")
            print(f"🧠 内存占用: {psutil.virtual_memory().percent}%")

        # 检查LR文件夹是否存在，如果不存在，则调用GenerateLRs函数生成LR图像
        if not os.path.exists(self.LRs_folder):
            print(f"LR文件夹不存在，正在生成LR图像...")
            # 创建LR存储目录
            bicubic_path = os.path.join(self.LRs_folder, 'bicubic')
            wavelet_path = os.path.join(self.LRs_folder, 'wavelet')
            gaussian_path = os.path.join(self.LRs_folder, 'Gaussian')
            os.makedirs(bicubic_path, exist_ok=True)
            if isTrain:
                os.makedirs(wavelet_path, exist_ok=True)
            #os.makedirs(gaussian_path, exist_ok=True)
    
            # 准备并行任务的参数
            tasks = [(hr_path, bicubic_path, wavelet_path, gaussian_path, 2, 'db1') 
                for hr_path in self.HRs_paths]
    
            # 并行执行
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(tqdm(
                    executor.map(generate_lr_single_image, tasks),
                    total=len(tasks),
                    desc="生成LR图像"
                ))
            print("LR图像生成完成！")

    def __len__(self):
        return len(self.HRs_paths)

    def __getitem__(self, idx):
        # 返回HRs, LRs和图像名称
        if self.preload:
            # 直接使用列表索引，检查索引是否有效，虽然 DataLoader 通常会保证
            if idx < 0 or idx >= len(self.HRs):
                 raise IndexError(f"索引 {idx} 超出范围 (0-{len(self.HRs)-1})")
            HR = self.HRs[idx].copy()
            img_name = self.img_names[idx]
        else:
            # 确保这里的 idx 对应 self.HRs_paths
            if idx < 0 or idx >= len(self.HRs_paths):
                 raise IndexError(f"索引 {idx} 超出范围 (0-{len(self.HRs_paths)-1})")
            img_path = self.HRs_paths[idx]
            # 添加检查以防文件在初始化后被删除或损坏
            if not os.path.exists(img_path):
                 raise FileNotFoundError(f"错误：图像文件在加载时丢失: {img_path}")
            try:
                HR_read = cv2.imread(img_path)
                if HR_read is None:
                    raise ValueError(f"错误：无法在 __getitem__ 中读取图像: {img_path}")
                HR = cv2.cvtColor(HR_read, cv2.COLOR_BGR2YCrCb)
                img_name = os.path.basename(img_path)
            except Exception as e:
                 raise IOError(f"错误：在 __getitem__ 中加载图像 {img_path} 时出错: {e}")

        # 确保 LR 图像存在
        LRs_folder_methods = [os.path.join(self.LRs_folder, f) for f in os.listdir(self.LRs_folder) if os.path.isdir(os.path.join(self.LRs_folder, f))]
        if not LRs_folder_methods:
            raise FileNotFoundError(f"错误：在 LR 文件夹 {self.LRs_folder} 中找不到任何下采样方法的子目录。")

        LR_methods = len(LRs_folder_methods)
        chosen_method_path = LRs_folder_methods[random.randint(0, LR_methods-1)]
        LR_path = os.path.join(chosen_method_path, img_name)

        if not os.path.exists(LR_path):
             raise FileNotFoundError(f"错误：对应的 LR 图像文件不存在: {LR_path}")

        try:
            LR_read = cv2.imread(LR_path)
            if LR_read is None:
                 raise ValueError(f"错误：无法读取 LR 图像: {LR_path}")
            LR = cv2.cvtColor(LR_read, cv2.COLOR_BGR2YCrCb)
        except Exception as e:
             raise IOError(f"错误：加载 LR 图像 {LR_path} 时出错: {e}")

        # 随机上下、左右翻转，旋转
        if self.isTrain:
            # 随机裁剪
            hr_height, hr_width, _ = HR.shape
            lr_height, lr_width, _ = LR.shape
            
            # 确保可以进行有效裁剪
            if hr_height >= self.block_size*self.scale_factor and hr_width >= self.block_size*self.scale_factor:
                # 计算LR图像中的对应块大小
                lr_block_size = self.block_size
                
                # 计算有效的随机范围
                max_x = lr_width - lr_block_size
                max_y = lr_height - lr_block_size
                
                if max_x >= 0 and max_y >= 0:
                    # 在LR图像上随机选择裁剪点
                    lr_x = random.randint(0, max_x)
                    lr_y = random.randint(0, max_y)
                    
                    # 计算HR图像上对应的裁剪点
                    hr_x = lr_x * self.scale_factor
                    hr_y = lr_y * self.scale_factor
                    
                    # 裁剪图像
                    HR = HR[hr_y:hr_y+self.block_size*self.scale_factor, hr_x:hr_x+self.block_size*self.scale_factor, :]
                    LR = LR[lr_y:lr_y+lr_block_size, lr_x:lr_x+lr_block_size, :]

            # 上下翻转
            if random.random() > 0.5:
                HR = cv2.flip(HR, 0)  
                LR = cv2.flip(LR, 0)

            # 左右翻转
            if random.random() > 0.5:
                HR = cv2.flip(HR, 1)  
                LR = cv2.flip(LR, 1)

            # 随机旋转
            if random.random() > 0.5:
                rotation_choices = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
                rotation_code = random.choice(rotation_choices)
                HR = cv2.rotate(HR, rotation_code)
                LR = cv2.rotate(LR, rotation_code)
        else:
            lr_height, lr_width, _ = LR.shape
            lr_height, lr_width = (lr_height//2)*2, (lr_width//2)*2
            LR = LR[0:lr_height, 0:lr_width, :]
            HR = HR[0:lr_height*self.scale_factor, 0:lr_width*self.scale_factor, :]
        LR = np.transpose(LR, (2, 0, 1))    # (C=3, H, W)
        HR = np.transpose(HR, (2, 0, 1))

        LR_Y = torch.from_numpy(LR[0:1]).float()
        LR_Cr = torch.from_numpy(LR[1:2]).float()
        LR_Cb = torch.from_numpy(LR[2:3]).float()

        HR_Y = torch.from_numpy(HR[0:1]).float()
        HR_Cr = torch.from_numpy(HR[1:2]).float()
        HR_Cb = torch.from_numpy(HR[2:3]).float()

        return LR_Y, LR_Cr, LR_Cb, HR_Y, HR_Cr, HR_Cb, img_name
    
def generate_lr_single_image(args):
    hr_path, bicubic_path, wavelet_path, gaussian_path, scale_factor, dwt_mode = args
    # 读取HR图像并转换为YCrCb格式
    HR = cv2.cvtColor(cv2.imread(hr_path), cv2.COLOR_BGR2YCrCb)
    
    # 计算LR_Cr, LR_Cb
    LR_Cr = cv2.resize(HR[:, :, 1], (HR.shape[1] // scale_factor, HR.shape[0] // scale_factor), interpolation=cv2.INTER_CUBIC)
    LR_Cb = cv2.resize(HR[:, :, 2], (HR.shape[1] // scale_factor, HR.shape[0] // scale_factor), interpolation=cv2.INTER_CUBIC)
    
    # 获取HR图像名称（包括后缀名）
    img_name = os.path.basename(hr_path)
    
    # 生成LR图像（方法1：直接下采样）
    LR_bicubic = cv2.resize(HR, (HR.shape[1] // scale_factor, HR.shape[0] // scale_factor), interpolation=cv2.INTER_CUBIC)
    LR_bicubic = cv2.cvtColor(LR_bicubic, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(os.path.join(bicubic_path, img_name), LR_bicubic)
    del LR_bicubic
    
    # 生成LR图像（方法2：使用小波变换的LL系数）
    LL, _ = pywt.dwt2(HR[:, :, 0], dwt_mode)
    LR_wt_Y = (LL / 2)[:, :, np.newaxis]  # LL系数作为Y通道
    LR_wt = np.concatenate((LR_wt_Y, LR_Cr[:, :, np.newaxis], LR_Cb[:, :, np.newaxis]), axis=2).astype(np.uint8)
    LR_wt = np.clip(LR_wt, 0, 255)
    cv2.imwrite(os.path.join(wavelet_path, img_name), cv2.cvtColor(LR_wt, cv2.COLOR_YCrCb2BGR))
    del LL, LR_wt_Y, LR_wt
    
    # 生成LR图像（方法3：使用高斯模糊）
    #LR_Gaussian = cv2.GaussianBlur(HR, (2, 2), 0)
    #cv2.imwrite(os.path.join(gaussian_path, img_name), cv2.cvtColor(LR_Gaussian, cv2.COLOR_YCrCb2BGR))
    #del LR_Gaussian

    return img_name

if __name__ == "__main__":
    # 测试数据加载器
    image_folder_train = os.path.join(os.path.dirname(__file__), 'data','train')
    train_data = MyImageDataset(image_folder=image_folder_train, transform=None, isTrain=True, preload=True, num_workers=4)
    image_folder_test = os.path.join(os.path.dirname(__file__), 'data','test')
    test_data = MyImageDataset(image_folder=image_folder_test, transform=None, isTrain=True, preload=True, num_workers=4)
