"""
StyTR-2 Transformer模块
实现了风格迁移的核心Transformer架构
包含编码器、解码器和注意力机制
用于智能融合内容和风格特征
"""

import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from function import normal, normal_style
import numpy as np
import os

# 设备配置
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"


class Transformer(nn.Module):
    """
    风格迁移Transformer主模块
    
    架构：
    1. 两个独立的Encoder（内容编码器 + 风格编码器）
    2. 一个Decoder（融合内容和风格）
    3. 内容感知的位置编码
    
    工作流程：
    - 内容图像通过encoder_c进行自注意力编码
    - 风格图像通过encoder_s进行自注意力编码
    - Decoder通过交叉注意力融合两者
    """

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        """
        参数:
            d_model: 模型的隐藏维度（默认512）
            nhead: 多头注意力的头数（默认8，即每个头64维）
            num_encoder_layers: 编码器层数（默认3）
            num_decoder_layers: 解码器层数（默认3）
            dim_feedforward: 前馈网络的隐藏层维度（默认2048）
            dropout: dropout比率
            activation: 激活函数类型
            normalize_before: 是否在attention前进行normalization
            return_intermediate_dec: 是否返回解码器的中间结果
        """
        super().__init__()

        # 创建编码器层的模板
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        
        # 创建两个独立的编码器
        # encoder_c: 处理内容图像
        # encoder_s: 处理风格图像
        # 注意：它们共享相同的架构，但参数独立
        self.encoder_c = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder_s = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # 创建解码器
        # 用于融合内容和风格特征
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        # 初始化参数
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        # 内容感知位置编码的生成网络
        self.new_ps = nn.Conv2d(512, 512, (1, 1))  # 1x1卷积
        self.averagepooling = nn.AdaptiveAvgPool2d(18)  # 自适应平均池化到18x18

    def _reset_parameters(self):
        """
        使用Xavier初始化参数
        这有助于训练的稳定性和收敛速度
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, style, mask, content, pos_embed_c, pos_embed_s):
        """
        前向传播 - 执行风格迁移的核心逻辑
        
        参数:
            style: 风格图像特征 [B, C, H, W]
            mask: 注意力掩码（用于处理不同大小的图像）
            content: 内容图像特征 [B, C, H, W]
            pos_embed_c: 内容的位置编码（会被重新计算，这里传入None）
            pos_embed_s: 风格的位置编码（会被重新计算，这里传入None）
        
        返回:
            融合后的特征 [B, C, H, W]
        """
        
        # ========== 步骤1: 生成内容感知的位置编码 ==========
        # 这是StyTR的创新点之一
        # 位置编码不是固定的，而是根据内容图像动态生成
        
        # 先将内容特征下采样到18x18
        content_pool = self.averagepooling(content)  # [B, 512, 18, 18]
        
        # 通过1x1卷积生成位置编码
        pos_c = self.new_ps(content_pool)  # [B, 512, 18, 18]
        
        # 将位置编码上采样到与风格特征相同的尺寸
        pos_embed_c = F.interpolate(pos_c, mode='bilinear', size=style.shape[-2:])
        # 现在 pos_embed_c 的尺寸与 style 相同

        # ========== 步骤2: 展平特征图为序列 ==========
        # Transformer处理序列数据，需要将2D特征图展平
        # 从 [B, C, H, W] -> [HW, B, C]
        
        # 展平风格特征
        # .flatten(2): 将H和W维度展平 -> [B, C, HW]
        # .permute(2, 0, 1): 调整维度顺序 -> [HW, B, C]
        style = style.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        
        if pos_embed_s is not None:
            pos_embed_s = pos_embed_s.flatten(2).permute(2, 0, 1)
        
        # 展平内容特征
        content = content.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        
        if pos_embed_c is not None:
            pos_embed_c = pos_embed_c.flatten(2).permute(2, 0, 1)
        
        # ========== 步骤3: 编码阶段 ==========
        # 分别对内容和风格进行自注意力编码
        # 目的：让每个位置理解自己在图像中的语义角色
        
        # 风格编码器：理解风格图像的结构
        # 例如：这个位置是天空、那个位置是树木
        style = self.encoder_s(style, src_key_padding_mask=mask, pos=pos_embed_s)
        # import pdb; pdb.set_trace()
        # 内容编码器：理解内容图像的结构
        # 例如：这个位置是人脸、那个位置是背景
        content = self.encoder_c(content, src_key_padding_mask=mask, pos=pos_embed_c)
        
        # ========== 步骤4: 解码阶段（关键！）==========
        # 通过交叉注意力融合内容和风格
        # content作为query（"我需要什么风格？"）
        # style作为key和value（"这里有哪些风格可用？"）
        hs = self.decoder(content, style, memory_key_padding_mask=mask,
                          pos=pos_embed_s, query_pos=pos_embed_c)[0]
        
        # ========== 步骤5: 重塑回特征图 ==========
        # 从序列格式 [HW, B, C] 恢复到特征图格式 [B, C, H, W]
        
        N, B, C = hs.shape  # N=HW, B=batch_size, C=channels
        H = int(np.sqrt(N))  # 假设H=W，计算高度
        
        # 调整维度顺序
        hs = hs.permute(1, 2, 0)  # [B, C, HW]
        
        # 重塑为2D特征图
        hs = hs.view(B, C, H, H)  # [B, C, H, H]（这里假设H=W）
        
        return hs


class TransformerEncoder(nn.Module):
    """
    Transformer编码器
    由多个TransformerEncoderLayer堆叠而成
    用于提取和增强特征的语义表示
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        """
        参数:
            encoder_layer: 单个编码器层（会被复制多次）
            num_layers: 编码器层的数量
            norm: 可选的layer normalization
        """
        super().__init__()
        # 深拷贝encoder_layer，创建多个独立的层
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """
        前向传播
        
        参数:
            src: 输入特征 [seq_len, batch_size, embed_dim]
            mask: 注意力掩码
            src_key_padding_mask: padding掩码
            pos: 位置编码
        
        返回:
            编码后的特征
        """
        output = src
        
        # 逐层处理
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        # 最后的layer normalization（如果有）
        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """
    Transformer解码器
    通过交叉注意力融合两个输入序列（内容和风格）
    这是实现风格迁移的关键模块
    """

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        """
        参数:
            decoder_layer: 单个解码器层
            num_layers: 解码器层数
            norm: layer normalization
            return_intermediate: 是否返回中间层的输出
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        前向传播 - 融合内容和风格
        
        参数:
            tgt: 目标序列（内容特征）[seq_len, batch, embed_dim]
            memory: 记忆序列（风格特征）[seq_len, batch, embed_dim]
            tgt_mask: 目标序列的注意力掩码
            memory_mask: 记忆序列的注意力掩码
            tgt_key_padding_mask: 目标序列的padding掩码
            memory_key_padding_mask: 记忆序列的padding掩码
            pos: 风格的位置编码
            query_pos: 内容的位置编码
        
        返回:
            融合后的特征（如果return_intermediate=True，返回所有中间结果的堆叠）
        """
        output = tgt

        intermediate = []

        # 逐层解码
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            
            # 如果需要，保存中间结果
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        # 最后的normalization
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        # 添加一个维度，保持输出格式一致
        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器的单层
    包含：自注意力 + 前馈网络
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        """
        参数:
            d_model: 模型维度
            nhead: 注意力头数
            dim_feedforward: 前馈网络隐藏层维度
            dropout: dropout比率
            activation: 激活函数
            normalize_before: normalization的位置（pre-norm或post-norm）
        """
        super().__init__()
        
        # 多头自注意力模块
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈网络（两层MLP）
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """
        将位置编码添加到特征中
        如果没有位置编码，直接返回原始特征
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        """
        Post-normalization的前向传播
        顺序：Attention -> Add -> Norm -> FFN -> Add -> Norm
        """
        # 自注意力
        # q和k都加上位置编码，v使用原始特征
        q = k = self.with_pos_embed(src, pos)
        
        # 计算自注意力
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        # 残差连接 + dropout
        src = src + self.dropout1(src2)
        # normalization
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        
        # 残差连接 + dropout
        src = src + self.dropout2(src2)
        # normalization
        src = self.norm2(src)
        
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """
        Pre-normalization的前向传播
        顺序：Norm -> Attention -> Add -> Norm -> FFN -> Add
        这种方式训练更稳定
        """
        # 先normalization
        src2 = self.norm1(src)
        
        # 自注意力
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        # 残差连接
        src = src + self.dropout1(src2)
        
        # 前馈网络
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """
        根据normalize_before选择前向传播方式
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器的单层
    包含：自注意力 + 交叉注意力 + 前馈网络
    
    交叉注意力是风格迁移的核心：
    - Query来自内容图像
    - Key和Value来自风格图像
    - 通过注意力机制，内容的每个位置会关注风格中最相关的部分
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        """
        参数:
            d_model: 模型维度（嵌入维度）
            nhead: 注意力头数
            dim_feedforward: 前馈网络隐藏层维度
            dropout: dropout比率
            activation: 激活函数类型
            normalize_before: 是否使用pre-normalization
        """
        super().__init__()
        
        # 自注意力：让内容特征理解自身
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 交叉注意力：内容和风格的融合（关键！）
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈网络（两层MLP）
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer Normalization（3个，对应3个子层）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """将位置编码添加到特征中"""
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """
        Post-normalization的前向传播
        
        参数:
            tgt: 目标序列（内容特征）
            memory: 记忆序列（风格特征）
            各种mask: 用于处理padding和特殊位置
            pos: 风格的位置编码
            query_pos: 内容的位置编码
        """
        
        # ========== 第一步：自注意力（可选，这里实际是交叉注意力）==========
        # 注意：这里的实现有点特殊
        # q来自内容，k来自风格，v也来自风格
        # 这实际上是在做交叉注意力，而非标准的自注意力
        q = self.with_pos_embed(tgt, query_pos)  # 内容 + 内容位置编码
        k = self.with_pos_embed(memory, pos)     # 风格 + 风格位置编码
        v = memory  # 风格特征（不加位置编码）
        
        # 计算注意力
        # 这里实现了：内容的每个位置去查询风格中最相关的部分
        tgt2 = self.self_attn(q, k, v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        
        # 残差连接 + dropout + norm
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # ========== 第二步：多头交叉注意力 ==========
        # 标准的交叉注意力：query来自tgt，key和value来自memory
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        
        # 残差连接 + dropout + norm
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # ========== 第三步：前馈网络 ==========
        # 两层MLP：d_model -> dim_feedforward -> d_model
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        
        # 残差连接 + dropout + norm
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        """
        Pre-normalization的前向传播
        先norm再进行计算，训练更稳定
        """
        # 第一步：自注意力（实际是交叉注意力）
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        
        # 第二步：交叉注意力
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        
        # 第三步：前馈网络
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        根据normalize_before选择前向传播方式
        """
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    """
    深拷贝模块N次
    用于创建多层Transformer
    
    参数:
        module: 要复制的模块
        N: 复制次数
    
    返回:
        包含N个独立模块的ModuleList
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    """
    根据参数构建Transformer
    这是一个工厂函数，方便从配置创建模型
    
    参数:
        args: 包含所有超参数的对象
    
    返回:
        配置好的Transformer模型
    """
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """
    根据字符串返回激活函数
    
    参数:
        activation: 激活函数的名称（'relu', 'gelu', 'glu'）
    
    返回:
        对应的激活函数
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
