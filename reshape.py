import cv2
import os
import glob

# 转为当前路径
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ✏️ 修改这里：输入/输出文件夹路径
input_dir = "compare/original"
output_dir = "compare/reshaped"
os.makedirs(output_dir, exist_ok=True)

# 存储矩形选区 (像素坐标)
roi = []
# 存储归一化坐标 [0,1] 范围
norm_roi = []
# 保存base_name
base_names=[]

def draw_rectangle(event, x, y, flags, param):
    global roi, display_img, hr_height, hr_width, drawing, original_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi = [(x, y)]
    
    elif event == cv2.EVENT_MOUSEMOVE:
        # 如果正在绘制中，实时显示矩形
        if drawing:
            # 计算正方形的第二个点
            start_x, start_y = roi[0]
            dx = x - start_x
            dy = y - start_y
            
            # 取绝对值较大的作为正方形边长
            side_length = max(abs(dx), abs(dy))
            
            # 根据拖动方向确定正方形的第二个点
            end_x = start_x + (side_length if dx > 0 else -side_length)
            end_y = start_y + (side_length if dy > 0 else -side_length)
            
            # 确保坐标在图像范围内
            end_x = min(max(end_x, 0), hr_width-1)
            end_y = min(max(end_y, 0), hr_height-1)
            
            # 复制原始图像，以便在上面绘制临时矩形
            temp_img = original_img.copy()
            cv2.rectangle(temp_img, roi[0], (end_x, end_y), (0, 255, 0), 2)
            display_img = temp_img
            cv2.imshow("Select ROI on HR", display_img)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # 计算正方形的第二个点
        start_x, start_y = roi[0]
        dx = x - start_x
        dy = y - start_y
        
        # 取绝对值较大的作为正方形边长
        side_length = max(abs(dx), abs(dy))
        
        # 根据拖动方向确定正方形的第二个点
        end_x = start_x + (side_length if dx > 0 else -side_length)
        end_y = start_y + (side_length if dy > 0 else -side_length)
        
        # 确保坐标在图像范围内
        end_x = min(max(end_x, 0), hr_width-1)
        end_y = min(max(end_y, 0), hr_height-1)
        
        roi.append((end_x, end_y))
        # 在最终图像上绘制矩形
        cv2.rectangle(display_img, roi[0], roi[1], (0, 255, 0), 2)
        cv2.imshow("Select ROI on HR", display_img)
        
        # 计算归一化坐标
        x1, y1 = roi[0]
        x2, y2 = roi[1]
        norm_roi.clear()
        norm_roi.append((x1/hr_width, y1/hr_height))
        norm_roi.append((x2/hr_width, y2/hr_height))

# 获取所有 *_HR.png 文件
hr_files = sorted(glob.glob(os.path.join(input_dir, "*_HR.png")))

if not hr_files:
    print("❌ 没有找到 *_HR.png 文件")
    exit()

for hr_path in hr_files:
    base_name = os.path.basename(hr_path).replace("_HR.png", "")
    base_names.append(base_name)  # 保存base_name
    lr_path = os.path.join(input_dir, f"{base_name}_LR.png")
    bicubic_path = os.path.join(input_dir, f"{base_name}_bicubic.png")
    Lanczos_path = os.path.join(input_dir, f"{base_name}_Lanczos.png")
    SR_dtcwt_path = os.path.join(input_dir, f"{base_name}_SR_dtcwt.png")
    SR_dwt_path = os.path.join(input_dir, f"{base_name}_SR_dwt.png")
    SRCNN_path = os.path.join(input_dir, f"{base_name}_SRCNN.png")

    if not all(os.path.exists(p) for p in [hr_path, lr_path, bicubic_path, Lanczos_path, SR_dtcwt_path, SR_dwt_path, SRCNN_path]):
        print(f"⚠️ 图组 {base_name} 不完整，跳过～")
        continue

    print(f"🎯 正在处理图组：{base_name}")

    # 加载图像
    images = {
        f"{base_name}_HR": cv2.imread(hr_path),
        f"{base_name}_LR": cv2.imread(lr_path),
        f"{base_name}_bicubic": cv2.imread(bicubic_path),
        f"{base_name}_dtcwt": cv2.imread(SR_dtcwt_path),
        f"{base_name}_dwt": cv2.imread(SR_dwt_path),
        f"{base_name}_SRCNN": cv2.imread(SRCNN_path),
        f"{base_name}_Lanczos": cv2.imread(Lanczos_path),
        
    }

    # 选择区域
    original_img = images[f"{base_name}_HR"].copy()  # 保存原始图像
    display_img = original_img.copy()
    hr_height, hr_width = display_img.shape[:2]
    roi = []
    norm_roi = []
    drawing = False  # 标记是否正在绘制

    cv2.namedWindow("Select ROI on HR", cv2.WINDOW_NORMAL)  # 使用WINDOW_NORMAL而不是默认的WINDOW_AUTOSIZE
    cv2.imshow("Select ROI on HR", display_img)

    # 如果图像太大，可以先调整窗口大小，保持图像比例
    h, w = display_img.shape[:2]
    max_dimension = 800  # 设置一个适当的最大尺寸
    scale_factor = min(1.0, max_dimension / max(h, w))
    resized_window_width = int(w * scale_factor)
    resized_window_height = int(h * scale_factor)
    cv2.resizeWindow("Select ROI on HR", resized_window_width, resized_window_height)

    cv2.setMouseCallback("Select ROI on HR", draw_rectangle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(roi) != 2 or len(norm_roi) != 2:
        print("⚠️ 未选择区域，跳过当前图组")
        continue
    
    # 创建并保存带有红色矩形框的HR图像
    os.makedirs(os.path.join(output_dir, base_name), exist_ok=True)
    hr_with_red_rect = original_img.copy()
    cv2.rectangle(hr_with_red_rect, roi[0], roi[1], (0, 0, 255), 5)  # BGR格式，红色是(0,0,255)
    red_rect_path = os.path.join(output_dir, base_name, f"{base_name}_HR_with_red_rect.png")
    cv2.imwrite(red_rect_path, hr_with_red_rect)
    print(f"✅ 已保存带红色矩形框的HR图像：{red_rect_path}")

    # 裁剪并保存
    for name, img in images.items():
        h, w = img.shape[:2]
        
        # 使用归一化坐标计算该图像上的实际像素坐标
        nx1, ny1 = norm_roi[0]
        nx2, ny2 = norm_roi[1]
        
        # 转换回像素坐标并确保在图像范围内
        x1, y1 = int(nx1 * w), int(ny1 * h)
        x2, y2 = int(nx2 * w), int(ny2 * h)
        
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        
        # 确保坐标在图像范围内
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)

        cropped = img[y_min:y_max, x_min:x_max]
        save_name = name + "_reshape.png"
        save_path = os.path.join(output_dir, base_name, save_name)
        cv2.imwrite(save_path, cropped)
        print(f"✅ 已保存：{save_path}")

    print("------ 下一组 ------\n")

print("🎉 全部图像处理完成～")

"""   写一个自动生成latex代码的程序，用于插入子图，子图的行*列=4*6，
 每一行的第一个图像是HR带红框的图像，第二个图像是HR裁剪后的图像，第三个图像是LR裁剪后的图像，
 第四个图像是bicubic裁剪后的图像，第五个图像是SR_dwt裁剪后的图像，第六个图像是SR_dtcwt裁剪后的图像，
 每一行的图片路径格式如下：
 比如某一行的文件名为base_name，那么该行的HR带框的图像路径为：base_name/base_name_HR_with_red_rect.png，
 HR裁剪后的图像路径为：base_name/base_name_HR_reshape.png，以此类推，
 每一列统一宽度为1/6，
 把生成的latex代码打印出来。 """

def generate_latex_subfigures(base_names):
    """
    生成LaTeX代码，用于插入4行6列的子图。
    
    Args:
        base_names: 包含4个基础名称的列表（对应4行）
    
    Returns:
        生成的LaTeX代码
    """
    latex_code = []
    
    # 添加LaTeX代码头部
    latex_code.append("\\begin{figure*}[htbp]")
    latex_code.append("\\centering")
    
    # 定义每列的标题
    column_titles = [
        "原图", 
        "HR", 
        "LR", 
        "双三次插值", 
        "DWT-SR", 
        "DTCWT-SR"
    ]
    
    # 添加图像行
    for base_name in base_names:
        image_paths = [
            f"{base_name}/{base_name}_HR_with_red_rect.png",
            f"{base_name}/{base_name}_HR_reshape.png",
            f"{base_name}/{base_name}_LR_reshape.png",
            f"{base_name}/{base_name}_bicubic_reshape.png",
            f"{base_name}/{base_name}_dwt_reshape.png",
            f"{base_name}/{base_name}_dtcwt_reshape.png"
        ]
        
        row_items = []
        for i, path in enumerate(image_paths):
            latex_code.append("\\begin{subfigure}[b]{0.16\\textwidth}")
            latex_code.append(f"\\includegraphics[width=0.16\\linewidth]{{{path}}}")
            if i == 0:
                subcaption = base_name + ".png"
            else:
                subcaption = column_titles[i]
            latex_code.append(f"\\caption{{{subcaption}}}")
            latex_code.append("\\label{fig:" + base_name + "_" + column_titles[i] + "}")
            latex_code.append("\\end{subfigure}")
            latex_code.append("\\hfill")
        latex_code.append(" ")
    
    # 添加LaTeX代码尾部
    latex_code.append("\\caption{对比不同超分辨率算法的视觉效果。}")
    latex_code.append("\\label{fig:visual_comparison}")
    latex_code.append("\\end{figure*}")
    
    return "\n".join(latex_code)

# 使用示例
latex_code = generate_latex_subfigures(base_names)
print(latex_code)
