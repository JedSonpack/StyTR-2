"""
StyTR-2 风格迁移测试脚本
用于将一张图片的艺术风格应用到另一张图片上
基于Transformer的神经风格迁移方法
"""

import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import models.transformer as transformer
import models.StyTR as StyTR
import matplotlib.pyplot as plt
from matplotlib import cm
from function import normal
import numpy as np
import time

# ========== 图像预处理函数 ==========

def test_transform(size, crop):
    """
    测试时的图像变换函数
    参数:
        size: 图像调整大小的目标尺寸
        crop: 是否进行中心裁剪
    返回:
        组合后的变换函数
    """
    transform_list = []
    
    # 如果指定了大小，添加调整大小的变换
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    # 如果需要裁剪，添加中心裁剪
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    # 将图像转换为张量（像素值范围从[0,255]转换到[0,1]）
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transform(h,w):
    """
    风格图像的变换函数
    参数:
        h: 目标高度
        w: 目标宽度
    返回:
        组合后的变换函数
    """
    k = (h,w)
    size = int(np.max(k))  # 取高度和宽度的最大值
    print(f"风格图像变换尺寸: {size}")
    transform_list = []    
    # 中心裁剪到指定尺寸
    transform_list.append(transforms.CenterCrop((h,w)))
    # 转换为张量
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    """
    内容图像的变换函数（仅转换为张量）
    返回:
        组合后的变换函数
    """
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

# ========== 命令行参数解析 ==========

parser = argparse.ArgumentParser(description='StyTR-2 风格迁移测试脚本')

# 基本选项
parser.add_argument('--content', type=str,
                    help='内容图像的文件路径（单张图片）')
parser.add_argument('--content_dir', type=str,
                    help='内容图像目录路径（批量处理）')
parser.add_argument('--style', type=str,
                    help='风格图像的文件路径（单张图片，多张用逗号分隔可做风格插值）')
parser.add_argument('--style_dir', type=str,
                    help='风格图像目录路径（批量处理）')
parser.add_argument('--output', type=str, default='output',
                    help='输出图像的保存目录（默认：output）')

# 模型路径参数
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth',
                    help='VGG特征提取器的权重文件路径')
parser.add_argument('--decoder_path', type=str, default='experiments/decoder_iter_160000.pth',
                    help='解码器的权重文件路径')
parser.add_argument('--Trans_path', type=str, default='experiments/transformer_iter_160000.pth',
                    help='Transformer的权重文件路径')
parser.add_argument('--embedding_path', type=str, default='experiments/embedding_iter_160000.pth',
                    help='嵌入层的权重文件路径')

# 高级选项
parser.add_argument('--style_interpolation_weights', type=str, default="",
                    help='风格插值权重（多个风格图像时使用）')
parser.add_argument('--a', type=float, default=1.0,
                    help='风格强度控制参数（0-1之间，1为最强）')
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="位置嵌入类型：sine（正弦）或learned（学习）")
parser.add_argument('--hidden_dim', default=512, type=int,
                    help="Transformer隐藏层维度（默认：512）")

# 解析命令行参数
args = parser.parse_args()

# ========== 参数验证 ==========

# 检查是否提供了必需的内容参数
if not args.content and not args.content_dir:
    print("错误：必须提供 --content 或 --content_dir 参数之一")
    print("示例：python test.py --content input/content/image.png --style input/style/style.png")
    exit(1)

# 检查是否提供了必需的风格参数
if not args.style and not args.style_dir:
    print("错误：必须提供 --style 或 --style_dir 参数之一")
    print("示例：python test.py --content input/content/image.png --style input/style/style.png")
    exit(1)

# ========== 高级参数设置 ==========

# 图像处理参数
content_size = 512      # 内容图像大小
style_size = 512        # 风格图像大小
crop = 'store_true'     # 是否裁剪（这里设置为字符串，实际应该是布尔值）
save_ext = '.jpg'       # 保存格式
output_path = args.output  # 输出路径
preserve_color = 'store_true'  # 是否保留颜色（未使用）
alpha = args.a          # 风格强度

# ========== 设备配置 ==========

# 检查是否有GPU可用，优先使用cuda:0，否则使用CPU
# 注意：原代码使用cuda:2，如果你的机器没有第3块GPU，会报错
if torch.cuda.is_available():
    # 检查可用的GPU数量
    gpu_count = torch.cuda.device_count()
    if gpu_count > 2:
        device = torch.device("cuda:2")
        print(f"使用设备: cuda:2")
    else:
        device = torch.device("cuda:0")
        print(f"使用设备: cuda:0 (共有{gpu_count}块GPU)")
else:
    device = torch.device("cpu")
    print("使用设备: CPU (未检测到GPU)")

# ========== 处理输入路径 ==========

# 处理内容图像路径
if args.content:
    # 单个内容图像
    content_paths = [Path(args.content)]
    print(f"处理单个内容图像: {args.content}")
else:
    # 内容图像目录
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    print(f"从目录加载 {len(content_paths)} 张内容图像")

# 处理风格图像路径
if args.style:
    # 单个风格图像
    style_paths = [Path(args.style)]
    print(f"处理单个风格图像: {args.style}")
else:
    # 风格图像目录
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    print(f"从目录加载 {len(style_paths)} 张风格图像")

# 创建输出目录（如果不存在）
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"创建输出目录: {output_path}")

# ========== 加载模型组件 ==========

print("\n开始加载模型...")

# 1. 加载VGG特征提取器
print("加载VGG特征提取器...")
vgg = StyTR.vgg
vgg.load_state_dict(torch.load(args.vgg))  # 加载预训练权重
vgg = nn.Sequential(*list(vgg.children())[:44])  # 只使用前44层

# 2. 创建模型各个组件
print("创建模型组件...")
decoder = StyTR.decoder              # 解码器：将特征转换回图像
Trans = transformer.Transformer()    # Transformer模块：进行风格迁移
embedding = StyTR.PatchEmbed()       # 图像块嵌入层：将图像分块并嵌入

# 3. 设置为评估模式（关闭dropout和batch normalization的训练行为）
decoder.eval()
Trans.eval()
vgg.eval()

# ========== 加载预训练权重 ==========

from collections import OrderedDict

# 加载解码器权重
print("加载解码器权重...")
new_state_dict = OrderedDict()
state_dict = torch.load(args.decoder_path)
for k, v in state_dict.items():
    # 如果权重文件包含'module.'前缀（多GPU训练产生），需要去除
    # namekey = k[7:] if k.startswith('module.') else k
    namekey = k
    new_state_dict[namekey] = v
decoder.load_state_dict(new_state_dict)

# 加载Transformer权重
print("加载Transformer权重...")
new_state_dict = OrderedDict()
state_dict = torch.load(args.Trans_path)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
Trans.load_state_dict(new_state_dict)

# 加载嵌入层权重
print("加载嵌入层权重...")
new_state_dict = OrderedDict()
state_dict = torch.load(args.embedding_path)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
embedding.load_state_dict(new_state_dict)

# ========== 构建完整网络 ==========

print("构建完整的风格迁移网络...")
# 将所有组件组合成完整的风格迁移网络
# 编码器 解码器 嵌入模块 Transformer模块
network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args)
network.eval()  # 设置为评估模式
network.to(device)  # 移动到指定设备（GPU或CPU）

print("模型加载完成！\n")

# ========== 准备图像变换 ==========

# 创建内容和风格图像的变换函数
content_tf = test_transform(content_size, crop)
style_tf = test_transform(style_size, crop)

# ========== 主处理循环 ==========

print(f"开始处理 {len(content_paths)} 张内容图像和 {len(style_paths)} 张风格图像...")
total_combinations = len(content_paths) * len(style_paths)
current = 0

# 对每个内容图像和风格图像的组合进行处理
for content_path in content_paths:
    for style_path in style_paths:
        current += 1
        print(f"\n[{current}/{total_combinations}] 正在处理:")
        print(f"  内容图像: {content_path.name}")
        print(f"  风格图像: {style_path.name}")
        
        # 加载并预处理内容图像
        content = content_tf(Image.open(content_path).convert("RGB"))
        
        # 获取内容图像的尺寸（注意：Tensor的维度顺序是 C,H,W）
        c, h, w = content.shape
        print(f"  内容图像尺寸: {h}x{w} (高x宽)")
        
        # 创建与内容图像尺寸匹配的风格变换
        style_tf1 = style_transform(h, w)
        
        # 加载并预处理风格图像
        style = style_tf(Image.open(style_path).convert("RGB"))
        
        # 将图像移动到设备并添加批次维度
        style = style.to(device).unsqueeze(0)  # 添加batch维度 [1, C, H, W]
        content = content.to(device).unsqueeze(0)  # 添加batch维度 [1, C, H, W]
        
        # 执行风格迁移（不计算梯度，节省内存）
        print("  执行风格迁移...")
        with torch.no_grad():
            output = network(content, style)
            # 网络可能返回元组 (output, loss_c, loss_s, l_identity1, l_identity2)
            # 测试时只需要第一个元素（风格化后的图像）
            if isinstance(output, tuple):
                output = output[0]
        
        # 将输出移回CPU
        output = output.cpu()
        
        # 生成输出文件名
        # 格式：output/内容图名_stylized_风格图名.jpg
        output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
            output_path, 
            splitext(basename(content_path))[0],  # 内容图文件名（不含扩展名）
            splitext(basename(style_path))[0],    # 风格图文件名（不含扩展名）
            save_ext  # 扩展名
        )
        
        # 保存风格化后的图像
        save_image(output, output_name)
        print(f"  已保存: {output_name}")

print(f"\n处理完成！共生成 {total_combinations} 张风格化图像")
print(f"输出目录: {output_path}")
