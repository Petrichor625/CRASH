import os
import copy
import math
import logging

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

    
# ==================================================================================================================== #
#                                                 Activation Functions                                                 #
# ==================================================================================================================== #
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

# ==================================================================================================================== #
#                                                     Gated layers                                                     #
# ==================================================================================================================== #

class BertGatedAttention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super(BertGatedAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        # 将输入向量拆分为多个头
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        # 注意力加权求和
        attended_values = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        # 经过线性变换和残差连接
        # x = self.fc(attended_values) + x
        
        return attended_values

class BertGatedIntermediate(nn.Module):
    def __init__(self,v_hidden_size):
        super(BertGatedIntermediate, self).__init__()
        self.has_vision = True

        # 自己设置
        self.v_hidden_size = v_hidden_size

        self.v_intermediate_size = self.v_hidden_size


        self.v_dense = nn.Linear(self.v_hidden_size, self.v_intermediate_size)
        self.v_intermediate_act_fn = ACT2FN['gelu']


    def forward(self, v_hidden_states):
        # v_hidden_states的形状是10,n,512
        v_hidden_states = self.v_dense(v_hidden_states)
        v_hidden_states = self.v_intermediate_act_fn(v_hidden_states)

        return v_hidden_states
    
class BertGatedFeedForward(nn.Module):
    def __init__(self,v_hidden_size):
        super(BertGatedFeedForward, self).__init__()

        self.v_hidden_size = v_hidden_size
        self.intermediate = BertGatedIntermediate(self.v_hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, v_input_tensor):
        v_inter_output = self.softmax(self.intermediate(v_input_tensor))
        return v_inter_output
    
    
class Encoder(nn.Module):
    # 需要两个参数，一个是隐藏层的大小，一个是注意力头的数量，子层都已经在下面定义过了
    def __init__(self,v_hidden_size,v_num_attention_heads):
        super(Encoder, self).__init__()


        # 初始化了两个集合：attn_sublayers 和 ff_sublayers，这些集合用于存储不同类型的自注意力子层和前馈神经网络子层的编号。
        # 在Set中+表示链接

        self.v_hidden_size = v_hidden_size
        self.v_num_attention_heads = v_num_attention_heads
       

        vv_attn_sublayers = [0,2,4]
        attn_sublayers = set(vv_attn_sublayers)
        
        v_ff_sublayers = [1,3,5]
        ff_sublayers = set(v_ff_sublayers)
       
        # 深度由attn和ffn共同决定
        depth = len(attn_sublayers) + len(ff_sublayers)
        # print("网络深度为：",depth)
        # 最后一步用到的，数量为层数+1，因为还有初始输入
        self.weights1 = nn.Parameter(torch.randn(len(attn_sublayers)+1))
        self.weights2 = nn.Parameter(torch.randn(len(attn_sublayers)+1))

        num2layers = {}
        self.num2type = {}
        # num2layer 字典存放的是真模型定义 
        # self.num2type 存放的是模型名字，在后续调用使用
        for n in attn_sublayers:
            num2layers[n] = BertGatedAttention(self.v_hidden_size,self.v_num_attention_heads)
            self.num2type[n] = "attn"

        for n in ff_sublayers:
            num2layers[n] = BertGatedFeedForward(self.v_hidden_size)
            self.num2type[n] = "ff"


        assert len(num2layers) == depth, "Overlapping attn-ff sublayer numbers"
        assert (min(num2layers) == 0) and (max(num2layers) == depth - 1), "Non contiguous sublayer numbers"

        self.layer1 = nn.ModuleList([copy.deepcopy(sublayer) for _, sublayer in sorted(num2layers.items())])
        self.layer2 = nn.ModuleList([copy.deepcopy(sublayer) for _, sublayer in sorted(num2layers.items())])

    def forward(self, h_stack):

        # print("h_stack的维度是：",h_stack.shape)
        h_first = h_stack[:, 0, :, :]  # 第一层，形状为N x 10 x 512
        h_first = h_first.permute(1, 0, 2) # 形状修改为10xNx512
        list_first = []
        list_first.append(h_first.mean(dim=1))
        h_second = h_stack[:, 1, :, :]  # 第二层，形状为N x 10 x 512
        h_second = h_second.permute(1, 0, 2) # 形状修改为10xNx512
        list_second = []
        list_second.append(h_second.mean(dim=1))

        for idx, layer in enumerate(self.layer1):
            layer_type = self.num2type[idx]
            # 把attn和ffn层交错排列，放在同一个list中间，依次抽取进行提取特征
            if layer_type == "attn":
                h_first = layer(h_first)
                # print("经过attn层，h_first的维度为：",h_first.shape) # 10,n,512
            else:   # ffn层
                h_first = layer(h_first) + h_first # 这里的输入其实是atten_values，形状为10,n,512
                list_first.append(h_first.mean(dim=1)) # 加入到列表中的都是10x512的张量
                # print("经过全连接层，h_first的维度为：",h_first.shape) # 10,n,512
        
        for idx, layer in enumerate(self.layer2):
            layer_type = self.num2type[idx]
            # 把attn和ffn层交错排列，放在同一个list中间，依次抽取进行提取特征
            if layer_type == "attn":
                h_second = layer(h_second)
            else:   # ffn层
                h_second = layer(h_second) + h_second
                list_second.append(h_second.mean(dim=1)) # 加入到列表中的都是10x512的张量

        # 都放到一起是为了充分利用每一层的信息，之和为1是为了训练稳定性
        weights_normalized1 = F.softmax(self.weights1, dim=0)
        aggregated_vector1 = torch.zeros_like(list_first[0])
        for i, vector in enumerate(list_first):
            aggregated_vector1 += weights_normalized1[i] * vector
        

        weights_normalized2 = F.softmax(self.weights2, dim=0)
        aggregated_vector2 = torch.zeros_like(list_second[0])
        for i, vector in enumerate(list_second):
            aggregated_vector2 += weights_normalized2[i] * vector
        
        
        h_stack = torch.stack([aggregated_vector1,aggregated_vector2], dim=0) # 2,10,512
        h_stack = F.softmax(h_stack, dim=-1)
        
        return h_stack