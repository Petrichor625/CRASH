'''
Muhammad Monjurul Karim

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
from timm.models.layers import DropPath, PatchEmbed
import math


from src.RSDlayerAttention import Encoder
from src.fft import SpectralGatingNetwork


class AccidentPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=2, act=torch.relu, dropout=[0, 0]):
        super(AccidentPredictor, self).__init__()
        self.act = act
        self.dropout = dropout
        self.dense1 = torch.nn.Linear(input_dim, 64)
        self.dense2 = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.dropout(x, self.dropout[0], training=self.training)
        x = self.act(self.dense1(x))
        x = F.dropout(x, self.dropout[1], training=self.training)
        x = self.dense2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_encoding', self.create_pos_encoding(d_model, max_len))

    def create_pos_encoding(self, d_model, max_len):
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        pos_encoding = pos_encoding.unsqueeze(0) 
        return pos_encoding

    def forward(self, x):
        return x + self.pos_encoding[:, :x.size(1)]

class SelfAttAggregate(nn.Module):
    def __init__(self, agg_dim, num_heads=4):
        super(SelfAttAggregate, self).__init__()
        self.agg_dim = agg_dim
        self.num_heads = num_heads
        self.pos_encoder = PositionalEncoding(512, 100)
        assert agg_dim % num_heads == 0, "agg_dim must be divisible by num_heads"

        self.depth = agg_dim // num_heads
        self.Wq = nn.Linear(agg_dim, agg_dim, bias=False)
        self.Wk = nn.Linear(agg_dim, agg_dim, bias=False)
        self.Wv = nn.Linear(agg_dim, agg_dim, bias=False)
        self.dense = nn.Linear(agg_dim, agg_dim)

        torch.nn.init.kaiming_normal_(self.Wq.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_normal_(self.Wk.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_normal_(self.Wv.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_normal_(self.dense.weight, a=math.sqrt(5))

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        return x

    def forward(self, hiddens):
        hiddens = hiddens.permute(0,2,1)
        hiddens = self.pos_encoder(hiddens)
        batch_size = hiddens.size(0)

        query = self.split_heads(self.Wq(hiddens), batch_size)
        key = self.split_heads(self.Wk(hiddens), batch_size)
        value = self.split_heads(self.Wv(hiddens), batch_size)

        matmul_qk = torch.matmul(query, key.transpose(-2, -1))
        depth = key.size(-1)
        logits = matmul_qk / math.sqrt(depth)
        weights = F.softmax(logits, dim=-1)

        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.agg_dim)

        output = self.dense(output)

        maxpool = torch.max(output, dim=1)[0]
        avgpool = torch.mean(output, dim=1)
        agg_feature = torch.cat((avgpool, maxpool), dim=1)

        return agg_feature




class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=[0,0]):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        input_dim = input_dim+512
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = dropout
        self.dense1 = torch.nn.Linear(hidden_dim, 64)
        self.dense2 = torch.nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = F.dropout(out[:,-1],self.dropout[0])
        out = self.relu(self.dense1(out))
        out = F.dropout(out,self.dropout[1])
        out = self.dense2(out)
        return out, h



class SpatialAttention(torch.nn.Module):

    def __init__(self, h_dim,n_layers):
        super (SpatialAttention, self).__init__()
        self.n_layers = n_layers
        self.q1 = nn.Linear(h_dim, h_dim)
        self.q2 = nn.Linear(h_dim, h_dim)
        self.wk = nn.Linear(h_dim, h_dim)
        self.wv = nn.Linear(h_dim, h_dim)
        self.alpha1 = nn.Parameter(torch.rand(1))
        self.alpha2 = nn.Parameter(torch.rand(1))

    def forward(self,obj_embed, h):
        query1 = self.q1(h[0]).unsqueeze(1) 
        query2 = self.q2(h[1]).unsqueeze(1)
        key = self.wk(obj_embed) 
        value = self.wv(obj_embed)
        attention_score1 = torch.bmm(query1,key.transpose(1,2))/math.sqrt(value.size(-1))
        attention_score2 = torch.bmm(query2,key.transpose(1,2))/math.sqrt(value.size(-1))
        attention_scores = self.alpha1*attention_score1 + self.alpha2*attention_score2
        attention_weights = F.softmax(attention_scores, dim=-1) 
        weighted_obj_embed = torch.bmm(attention_weights, value)
        return weighted_obj_embed

class CRASH(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers=1, n_obj=19, n_frames=100, fps=20.0, with_saa=True):
        super(CRASH, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.n_obj = n_obj
        self.n_frames = n_frames
        self.fps = fps
        self.with_saa = with_saa
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        self.phi_x3 = nn.Sequential(nn.Linear(512, 512), nn.ReLU()) 
        self.sp_attention = SpatialAttention(self.h_dim,self.n_layers)
        self.rho_1 = torch.nn.Parameter(torch.tensor(1.0))
        self.rho_2 = torch.nn.Parameter(torch.tensor(1.0))
        self.gru_net = GRUNet(h_dim+h_dim , h_dim, 2, n_layers,dropout=[0.5, 0.0])
        if self.with_saa:
            self.predictor_aux = AccidentPredictor(h_dim + h_dim, 2, dropout=[0.5, 0.0])
            self.self_aggregation = SelfAttAggregate(self.h_dim)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        
        self.RSD = Bert_VL_Encoder(512,8)
        self.fftblock = SpectralGatingNetwork(3)


    def forward(self, x, y, toa, hidden_in=None, nbatch=80, testing=False):
        losses = {'cross_entropy': 0,
                  'total_loss': 0,
                  'log' : 0}
        if self.with_saa:
            losses.update({'auxloss': 0})
        all_outputs, all_hidden = [], []


        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layers, x.size(0),  self.h_dim))
        else:
            h = Variable(hidden_in)
        h = h.to(x.device)


        h_list = []

        for t in range(x.size(1)):
            x_t = self.phi_x(x[:, t])
            img_embed = x_t[:, 0, :].unsqueeze(1)
            img_tmp = img_embed.view(x.size(0),512,1)
            img_tmp = self.fftblock(img_tmp)
            img_tmp = img_tmp.view(x.size(0),1,512)
            img_fft = self.phi_x3(img_tmp)
            obj_embed = x_t[:, 1:, :]
            obj_embed= self.sp_attention(obj_embed, h)
            x_t = torch.cat([obj_embed, img_embed,img_fft], dim=-1)
            h_list.append(h)

            if t==2:
                h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2]),dim=0)
                h = self.RSD(h_staked)
            elif t==3:
                h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2], h_list[t-3]),dim=0)
                h = self.RSD(h_staked)
            elif t==4:
                h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2], h_list[t-3], h_list[t-4]),dim=0)
                h = self.RSD(h_staked)
            elif t==5:
                h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2], h_list[t-3], h_list[t-4], h_list[t-5]),dim=0)
                h = self.RSD(h_staked)
            elif t==6:
                h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2], h_list[t-3], h_list[t-4], h_list[t-5], h_list[t-6]),dim=0)
                h = self.RSD(h_staked)
            elif t==7:
                h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2], h_list[t-3], h_list[t-4], h_list[t-5], h_list[t-6], h_list[t-7]),dim=0)
                h = self.RSD(h_staked)
            elif t==8:
                h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2], h_list[t-3], h_list[t-4], h_list[t-5], h_list[t-6], h_list[t-7], h_list[t-8]),dim=0)
                h = self.RSD(h_staked)
            elif t>8:
                h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2], h_list[t-3], h_list[t-4], h_list[t-5], h_list[t-6], h_list[t-7], h_list[t-8], h_list[t-9]),dim=0)
                h = self.RSD(h_staked)
            output, h = self.gru_net(x_t, h)

            L3 = self._exp_loss(output, y, t, toa=toa, fps=self.fps)
            losses['cross_entropy'] += L3
            all_outputs.append(output)
            all_hidden.append(h[-1])

        if self.with_saa:
            embed_video = self.self_aggregation(torch.stack(all_hidden, dim=-1))
            dec = self.predictor_aux(embed_video)
            L4 = torch.mean(self.ce_loss(dec, y[:, 1].to(torch.long)))
            L4 = L4 / (self.rho_2 * self.rho_2)
            losses['auxloss'] = L4
        
        losses['log'] = torch.log(self.rho_1 * self.rho_2)

        return losses, all_outputs, all_hidden


    def _exp_loss(self, pred, target, time, toa, fps=10.0):
        target_cls = target[:, 1]
        target_cls = target_cls.to(torch.long)
        penalty = - 0.5 * torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), (toa.to(pred.dtype) - time - 1) / fps)
        pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
        neg_loss = self.ce_loss(pred, target_cls)
        loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
        loss = loss / (self.rho_1 * self.rho_1)
        return loss