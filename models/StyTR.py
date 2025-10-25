"""
StyTR-2 风格迁移模型核心模块
实现了基于Transformer的神经风格迁移网络
包含图像编码器(VGG)、解码器、图像块嵌入和Transformer模块
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from function import normal,normal_style
from function import calc_mean_std
import scipy.stats as stats
from models.ViT_helper import DropPath, to_2tuple, trunc_normal_

# ========== 图像块嵌入模块 ==========

class PatchEmbed(nn.Module):
    """
    图像到补丁嵌入 (Image to Patch Embedding)
    将输入图像分割成补丁(patches)并投影到嵌入空间
    这是Vision Transformer的关键步骤
    """
    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        """
        参数:
            img_size: 输入图像大小 (默认256x256)
            patch_size: 每个补丁的大小 (默认8x8)
            in_chans: 输入通道数 (RGB图像为3)
            embed_dim: 嵌入维度 (默认512)
        """
        super().__init__()
        img_size = to_2tuple(img_size)  # 转换为元组 (256, 256)
        patch_size = to_2tuple(patch_size)  # 转换为元组 (8, 8)
        
        # 计算补丁数量: (256/8) * (256/8) = 32 * 32 = 1024个补丁
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # 使用卷积进行补丁嵌入
        # 卷积核大小和步长都等于patch_size，这样可以将图像分成不重叠的补丁
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 上采样层（2倍）
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        前向传播
        输入: [B, 3, H, W] - 批次、通道、高度、宽度
        输出: [B, 512, H/8, W/8] - 嵌入后的特征图
        """
        B, C, H, W = x.shape
        # 通过卷积将图像投影到嵌入空间
        x = self.proj(x)
        return x


# ========== 解码器网络 ==========
# 将特征图解码回RGB图像
# 通过逐步上采样和卷积，从512维特征恢复到3通道图像

decoder = nn.Sequential(
    # 第一阶段: 512 -> 256 通道
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 反射填充，避免边界伪影
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),  # 2倍上采样
    
    # 第二阶段: 256通道的深度卷积
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    
    # 第三阶段: 256 -> 128 通道
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),  # 2倍上采样
    
    # 第四阶段: 128通道的卷积
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    
    # 第五阶段: 128 -> 64 通道
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),  # 2倍上采样
    
    # 第六阶段: 64通道的卷积
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    
    # 最后阶段: 64 -> 3 通道 (RGB)
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),  # 输出RGB图像
)

# ========== VGG特征提取器 ==========
# 预训练的VGG网络，用于提取图像的内容和风格特征
# 包含5个主要层级（relu1 到 relu5）

vgg = nn.Sequential(
    # 预处理层
    nn.Conv2d(3, 3, (1, 1)),
    
    # ===== 第一层级 (relu1) =====
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 2倍下采样
    
    # ===== 第二层级 (relu2) =====
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 2倍下采样
    
    # ===== 第三层级 (relu3) =====
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 2倍下采样
    
    # ===== 第四层级 (relu4) ===== 
    # 这是风格迁移中最常用的特征层
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, 这是主要使用的特征层
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 2倍下采样
    
    # ===== 第五层级 (relu5) =====
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

# ========== 多层感知机 ==========

class MLP(nn.Module):
    """
    非常简单的多层感知机 (也称为前馈神经网络 FFN)
    用于Transformer中的前馈层
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: 网络层数
        """
        super().__init__()
        self.num_layers = num_layers
        
        # 创建隐藏层维度列表
        h = [hidden_dim] * (num_layers - 1)
        
        # 创建多层线性层
        # 例如: input_dim -> hidden_dim -> hidden_dim -> output_dim
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        """
        前向传播
        除最后一层外，每层后面都接ReLU激活函数
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# ========== 风格迁移Transformer主模块 ==========

class StyTrans(nn.Module):
    """
    风格迁移Transformer模块
    这是StyTR-2的核心网络，负责将风格图像的风格迁移到内容图像上
    
    主要流程:
    1. 使用VGG编码器提取内容和风格图像的多层特征
    2. 使用PatchEmbed将特征转换为补丁嵌入
    3. 通过Transformer融合内容和风格特征
    4. 使用解码器将融合后的特征解码为风格化图像
    5. 计算多个损失函数以优化网络
    """
    
    def __init__(self, encoder, decoder, PatchEmbed, transformer, args):
        """
        参数:
            encoder: VGG编码器，用于特征提取
            decoder: 解码器，将特征转换回图像
            PatchEmbed: 补丁嵌入层
            transformer: Transformer模块，进行特征融合
            args: 其他参数
        """
        super().__init__()
        
        # 将VGG编码器分成5个层级
        # 每个层级对应不同尺度的特征
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])    # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18]) # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31]) # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44]) # relu4_1 -> relu5_1
        
        # 冻结编码器参数，只训练Transformer和解码器
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        # 损失函数
        self.mse_loss = nn.MSELoss()  # 均方误差损失
        
        # 其他模块
        self.transformer = transformer
        hidden_dim = transformer.d_model  # Transformer的隐藏维度
        self.decode = decoder
        self.embedding = PatchEmbed

    def encode_with_intermediate(self, input):
        """
        使用VGG编码器提取多层中间特征
        
        参数:
            input: 输入图像 [B, 3, H, W]
        
        返回:
            包含5层特征的列表，对应relu1到relu5
        """
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]  # 返回5层特征，不包括原始输入

    def calc_content_loss(self, input, target):
        """
        计算内容损失
        确保生成图像保留原始内容图像的内容结构
        
        参数:
            input: 生成图像的特征
            target: 目标内容图像的特征
        
        返回:
            内容损失值
        """
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        """
        计算风格损失
        通过匹配特征的均值和标准差来迁移风格
        
        原理: 风格可以通过特征的统计信息（均值和方差）来表示
        
        参数:
            input: 生成图像的特征
            target: 目标风格图像的特征
        
        返回:
            风格损失值
        """
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        
        # 计算均值和标准差
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        
        # 同时匹配均值和标准差
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    def forward(self, samples_c: NestedTensor, samples_s: NestedTensor):
        """
        前向传播 - 执行风格迁移
        
        参数:
            samples_c: 内容图像 (NestedTensor格式)
                - samples_c.tensor: 图像张量 [batch_size, 3, H, W]
                - samples_c.mask: 填充掩码 [batch_size, H, W]
            samples_s: 风格图像 (NestedTensor格式)
        
        返回:
            训练模式: (Ics, loss_c, loss_s, loss_lambda1, loss_lambda2)
                - Ics: 风格化后的图像
                - loss_c: 内容损失
                - loss_s: 风格损失
                - loss_lambda1: 身份损失1
                - loss_lambda2: 身份损失2
            测试模式: Ics (仅返回风格化图像)
        """
        # 保存原始输入（用于计算身份损失）
        content_input = samples_c
        style_input = samples_s
        
        # 如果输入是列表或张量，转换为NestedTensor格式
        # NestedTensor支持不同尺寸的图像（通过padding和mask）
        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(samples_c)
        if isinstance(samples_s, (list, torch.Tensor)):
            samples_s = nested_tensor_from_tensor_list(samples_s)
        
        # ========== 特征提取 ==========
        # 提取内容和风格图像的多层VGG特征（用于计算损失）
        content_feats = self.encode_with_intermediate(samples_c.tensors)
        style_feats = self.encode_with_intermediate(samples_s.tensors)

        # ========== 线性投影 ==========
        # 将图像转换为补丁嵌入
        style = self.embedding(samples_s.tensors)
        content = self.embedding(samples_c.tensors)
        
        # 位置编码（在transformer.py中计算）
        pos_s = None  # 风格图像的位置编码
        pos_c = None  # 内容图像的位置编码
        mask = None   # 注意力掩码

        # ========== Transformer风格迁移 ==========
        # 使用Transformer融合内容和风格
        # 输入: 风格特征作为query，内容特征作为key和value
        hs = self.transformer(style, mask, content, pos_c, pos_s)
        
        # ========== 解码 ==========
        # 将融合后的特征解码为RGB图像
        Ics = self.decode(hs)

        # ========== 计算损失 ==========
        
        # 1. 内容损失 - 确保保留内容结构
        # 使用relu4-1和relu5-1层的特征
        Ics_feats = self.encode_with_intermediate(Ics)
        loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1])) + \
                 self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))
        
        # 2. 风格损失 - 迁移风格特征
        # 使用所有5层特征的统计信息
        loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])
        
        # 3. 身份损失1 - 确保内容->内容和风格->风格的重建
        # Icc: 内容图像经过网络后应该保持不变
        # Iss: 风格图像经过网络后应该保持不变
        Icc = self.decode(self.transformer(content, mask, content, pos_c, pos_c))
        Iss = self.decode(self.transformer(style, mask, style, pos_s, pos_s))
        loss_lambda1 = self.calc_content_loss(Icc, content_input) + \
                      self.calc_content_loss(Iss, style_input)
        
        # 4. 身份损失2 - 在特征层面确保一致性
        Icc_feats = self.encode_with_intermediate(Icc)
        Iss_feats = self.encode_with_intermediate(Iss)
        loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0]) + \
                      self.calc_content_loss(Iss_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i]) + \
                           self.calc_content_loss(Iss_feats[i], style_feats[i])
        
        # ========== 返回结果 ==========
        # 训练模式：返回图像和所有损失
        # 测试模式：只返回风格化图像（需要注释掉训练模式的返回语句）
        return Ics, loss_c, loss_s, loss_lambda1, loss_lambda2  # 训练模式
        # return Ics  # 测试模式（取消注释此行，注释上一行）
