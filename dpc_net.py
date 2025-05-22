#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Xjr
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import os
import sys
import copy
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointlstm import PointLSTM

# devices = 'cuda:0'
PT_NUM = 1024
enhance_frame = 2 # 基准帧，最后一帧-1或中间帧1


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    dis = pairwise_distance.topk(k=k, dim=-1)[0]
    return idx, dis


def PCA(data, k, sort=True):       # (batch_size, 2048, 20, 3)
    data_mean = torch.mean(data, dim=-2, keepdim=True)
    channel = data.size()[-1]
    batch_size = data.size()[0]
    data_normalize = data - data_mean.repeat(1, 1, k, 1)
    data_normalize = data_normalize.view(-1, k, channel)
    h = torch.matmul(data_normalize.transpose(2, 1), data_normalize)
    eigenvectors, _, _ = torch.svd(h)   # batch_size*2048, 3
    eigenvectors = eigenvectors.view(batch_size, -1, channel, channel)
    return eigenvectors


def Normal(data, idx, k, devices):
    batch_size, num_points, num_dims = data.size()
    # num_points = data.size()[1]
    # num_dims = data.size()[2]

    idx_base = torch.arange(0, batch_size, device=torch.device(devices)).view(-1, 1, 1) * num_points
    idx1 = idx + idx_base
    idx1 = idx1.view(-1)

    k_nearest_point = data.view(batch_size * num_points, -1)[idx1, :]
    k_nearest_point = k_nearest_point.view(batch_size, num_points, k, num_dims)
    # 存储法向量
    u = PCA(k_nearest_point, k)    # (batch_size, 2048, 3, 3)
    normals = torch.squeeze(u[:, :, :, -1])        # (batch_size, 2048, 3)
    normals = nn.functional.normalize(normals, 2, dim=-1)
    return normals


def get_graph_feature(x, pos, idx=None, dis=None, k=20, devices='cuda:0'):
    batch_size = x.size(0)
    num_points = x.size(2)

    x = x.contiguous().view(batch_size, -1, num_points)
    pos = pos.contiguous().view(batch_size, -1, num_points)
    if idx is None:
        idx, dis = knn(pos, k=k)  # (batch_size, num_points, k)
        dis = torch.unsqueeze(dis, dim=-3)
    device = torch.device(devices)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx1 = idx + idx_base
    idx1 = idx1.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx1, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature_edge = (feature - x).permute(0, 3, 1, 2).contiguous()
    feature_edge_mix = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature_edge, feature_edge_mix, idx, dis


class GAPLayer_single(nn.Module):
    def __init__(self, p_num, k=20, channel=16, inputsize=1, devices='cuda:0'):
        super(GAPLayer_single, self).__init__()
        self.p_num = p_num
        self.k = k
        self.channel = channel
        self.devices = devices
        self.conv1 = nn.Sequential(nn.Conv2d(2 * inputsize, channel, 1, bias=False),
                                   nn.BatchNorm2d(channel),
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.convy = nn.Sequential(nn.Conv2d(2 * inputsize, channel, 1, bias=False),
        #                            nn.BatchNorm2d(channel),
        #                            nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(channel, 1, 1, bias=False),
                                   nn.BatchNorm2d(1))

        self.conv1n = nn.Sequential(nn.Conv2d(2 * inputsize, channel, 1, bias=False),
                                    nn.BatchNorm2d(channel),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.conv2n = nn.Sequential(nn.Conv2d(channel, 1, 1, bias=False),
                                    nn.BatchNorm2d(1))
        self.LR = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pos, idx, dis):
        _, edge2, _, _ = get_graph_feature(x, pos, idx, dis, devices=self.devices)

        x = self.conv1(edge2)  # (batch, channel, p_num, 1)
        edge_o = self.conv1n(edge2)  # (batch, channel, p_num, k)

        x = self.conv2(x)  # (batch, 1, p_num, 1)
        edge = self.conv2n(edge_o)  # (batch, 1, p_num, k)
        edge_o = edge_o.permute(0, 2, 3, 1).contiguous()  # batch, p_num, k, channel

        x = self.LR(x + edge)
        x = self.softmax(x)
        x = x.permute(0, 2, 1, 3).contiguous()  # batch, p_num, 1, k
        # print(x.size(), edge_o.size())
        x = torch.matmul(x, edge_o)
        x = torch.squeeze(x)

        return x, edge_o


class GAPLayer(nn.Module):
    def __init__(self, channel=16, inputsize=1, devices='cuda:0'):
        super(GAPLayer, self).__init__()
        dim = PT_NUM
        k = 20
        self.single1 = GAPLayer_single(dim, k, channel, inputsize, devices=devices)
        self.single2 = GAPLayer_single(dim, k, channel, inputsize, devices=devices)
        # self.single3 = GAPLayer_single(dim, k, channel, inputsize)
        # self.single4 = GAPLayer_single(dim, k, channel, inputsize)
        self.single3 = GAPLayer_single(dim, k, 2 * channel, 2 * channel, devices=devices)

    def forward(self, x, pos, idx, dis):
        x1, xn1 = self.single1(x, pos, idx, dis)
        x2, xn2 = self.single2(x, pos, idx, dis)
        x = torch.cat((x1, x2), dim=-1)     # batch_size*p_num*32
        xn_o = torch.cat((xn1, xn2), dim=-1)    # batch_size*p_num*k*32
        x_t = torch.unsqueeze(x, dim=-1)
        x_t = x_t.permute(0, 2, 1, 3).contiguous()
        x3, xn3 = self.single3(x_t, pos, idx, dis)
        x = torch.cat((x, x3), dim=-1)       # batch_size*p_num*64
        xn_o = torch.cat((xn_o, xn3), dim=-1)       # batch_size*p_num*k*64

        # x3, xn3 = self.single3(x, pos, idx, dis, normal)
        # x4, xn4 = self.single4(x, pos, idx, dis, normal)
        # x = torch.cat((x1, x2, x3, x4), dim=-1)  # batch_size*p_num*64
        # xn_o = torch.cat((xn1, xn2, xn3, xn4), dim=-1)  # batch_size*p_num*k*64
        return x, xn_o


class GAPCN(nn.Module):
    def __init__(self, devices):
        super(GAPCN, self).__init__()
        self.k = 20
        self.devices = devices

        self.GAP1 = GAPLayer(channel=16, inputsize=1, devices=self.devices) # 单通道训练为1, yuv为3
        self.GAP2 = GAPLayer(channel=64, inputsize=64, devices=self.devices) # 64

        self.conv1 = nn.Sequential(nn.Conv2d(133, 128, 1, bias=False), nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(576, 256, 1, bias=False), nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(256, 128, 1, bias=False), nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(128, 1, 1, bias=False))

        self.conv3 = nn.Sequential(nn.Conv2d(643, 192, 1, bias=False), nn.BatchNorm2d(192),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(192, 128, 1, bias=False), nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(128, 256, 1, bias=False), nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.convn2 = nn.Sequential(nn.Conv2d(256, 128, 1, bias=False), nn.BatchNorm2d(128),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv2d(128, 256, 1, bias=False), nn.BatchNorm2d(256),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.convn1 = nn.Sequential(nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64),
                                    nn.LeakyReLU(negative_slope=0.2))
        # self.atten1 = self_attention(channel=130)
        # self.atten2 = self_attention(channel=640)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, qp=0):
        x_pos = x[:, :3, :]
        x = x[:, 3:, :]       # batch_size, 1, num_points
        rec = x.permute(0, 2, 1).contiguous()        # batch_size, num_points, 3
        feature, _, idx, dis = get_graph_feature(x, x_pos, devices=self.devices)  # batch_size, num_points, k, 1
        normal = Normal(x_pos.transpose(2, 1).contiguous(), idx, self.k, devices=self.devices)  # batch_size, num_points, 3
        normal = normal.transpose(2, 1).contiguous()  # batch_size, 3, num_points
        normals, _, idx, dis = get_graph_feature(normal, x_pos, devices=self.devices)     # batch_size, num_points, k, 3
        normals = normals.permute(0, 3, 1, 2).contiguous()

        dis_k = 2 * (torch.ones_like(dis) - self.sigmoid(dis))

        x1, xn = self.GAP1(x, x_pos, idx, dis)
        x = x.permute(0, 2, 1).contiguous()

        x = torch.cat((x, x1), dim=-1)  # batch_size*p_num*65
        x = x.permute(0, 2, 1).contiguous()  # batch_size*65*p_num
        _, x, _, _ = get_graph_feature(x, x_pos, idx, devices=self.devices)  # batch_size*130*p_num*k
        dises = dis_k.repeat(1, 130, 1, 1)  # (batch, 130, p_num, k)
        x = torch.mul(x, dises)
        normals = normals.permute(0, 2, 3, 1).contiguous()
        # print('normals shape:', normals.shape) # [2, 3, 2048, 20]
        x = torch.cat((x, normals), dim=-3)
        x = self.conv1(x)
        x = torch.max(x, dim=-1, keepdim=False)[0]  # batch_size*128*p_num
        # x = torch.mean(x, dim=-1, keepdim=False)  # batch_size*128*p_num

        x = x.transpose(2, 1).contiguous()  # batch_size*p_num*64
        x2 = x.transpose(2, 1).contiguous()

        x2, xn2 = self.GAP2(x2, x_pos, idx, dis)  # batch_size*p_num*256, batch_size*p_num*k*256

        xn2 = xn2.permute(0, 3, 1, 2).contiguous()  # batch_size*256*p_num*k
        xn2 = self.convn2(xn2)  # batch_size*256*p_num*k
        xn2 = torch.max(xn2, dim=-1, keepdim=False)[0]  # batch_size*256*p_num
        # xn = torch.mean(xn, dim=2, keepdim=True)
        xn2 = xn2.transpose(2, 1).contiguous()      # batch_size*p_num*256

        xn = xn.permute(0, 3, 1, 2).contiguous()    # batch_size*64*p_num*k
        xn = self.convn1(xn)                        # batch_size*64*p_num*k
        xn = torch.max(xn, dim=-1, keepdim=False)[0]    # batch_size*64*p_num
        # xn = torch.mean(xn, dim=2, keepdim=True)
        xn = xn.transpose(2, 1).contiguous()

        x2 = torch.cat((x2, x), dim=-1)  # batch_size*p_num*320
        x2 = x2.permute(0, 2, 1).contiguous()  # batch_size*320*p_num
        _, x2, _, _ = get_graph_feature(x2, x_pos, idx, devices=self.devices)
        dises = dis_k.repeat(1, 640, 1, 1)  # (batch, 640, p_num, k)
        x2 = torch.mul(x2, dises)
        # print('x2 shape:', x2.shape) # [2, 640, 2048, 20]
        # print('normals shape:', normals.shape) # [2, 3, 2048, 20]
        x2 = torch.cat((x2, normals), dim=-3)
        x2 = self.conv3(x2)  # batch_size*256*p_num*k
        x2 = torch.max(x2, dim=-1, keepdim=False)[0]  # batch_size*256*p_num
        # x2 = torch.mean(x2, dim=-1, keepdim=False)  # batch_size*256*p_num
        x2 = x2.permute(0, 2, 1).contiguous()  # batch_size*p_num*256

        x = torch.cat((xn, x2, xn2), dim=-1)  # batch_size*p_num*(256+256+64+64) = 640

        # x = torch.unsqueeze(x, dim=-1)
        # x = x.permute(0, 2, 1, 3).contiguous()  # batch_size*640*p_num*1
        # x = self.conv2(x)

        # x = torch.squeeze(x)
        # x = torch.unsqueeze(x, dim=-2)
        # x = x.transpose(2, 1)  # batch_size*p_num*1
        # x = x + rec

        return x


class dpc(nn.Module):
    def __init__(self, devices='cuda:0'):
        super(dpc, self).__init__()
        self.devices = devices
        self.GAPCN = GAPCN(devices)

        # self.lstm1 = PointLSTM(offsets=False, pts_num=PT_NUM, in_channels=576, hidden_dim=32,
        #                       offset_dim=4, num_layers=1, topk=16)
        
        self.lstm1 = PointLSTM(offsets=False, pts_num=PT_NUM, in_channels=576, hidden_dim=64,
                              offset_dim=4, num_layers=1, topk=16)
        
        # self.conv = nn.Sequential(nn.Conv2d(32, 16, 1, bias=False), nn.BatchNorm2d(16),
        #                            nn.LeakyReLU(negative_slope=0.2),
        #                            nn.Conv2d(16, 8, 1, bias=False), nn.BatchNorm2d(8),
        #                            nn.LeakyReLU(negative_slope=0.2),
        #                            nn.Conv2d(8, 1, 1, bias=False))
        
        self.conv = nn.Sequential(nn.Conv2d(640, 256, 1, bias=False), nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(256, 128, 1, bias=False), nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(128, 1, 1, bias=False))

    def forward(self, x):
        # batch_size, num_frames, _, num_points = x.size()
        rec = x[:, enhance_frame, 3:, :].permute(0, 2, 1).contiguous() # batch_size, num_points, 3

# ------------------------------------ 原始GAPCN ------------------------------------
        # x = self.GAPCN1(x[:, enhance_frame, :, :512], self.devices)
        # x = torch.unsqueeze(x, dim=-1)
        # x = x.permute(0, 2, 1, 3).contiguous()  # batch_size*640*p_num*1
        # x = self.conv(x)
        # x = torch.squeeze(x)
        # x = torch.unsqueeze(x, dim=-2)
        # x = x.transpose(2, 1)  # batch_size*p_num*1
        # x = x + rec
# ------------------------------------ 原始GAPCN ------------------------------------


# ------------------------------------------- PlanA -------------------------------------------
        # feat1 = self.GAPCN(x[:, 0, :, :], self.devices) # [batchsize, 2048, 576]
        # feat2 = self.GAPCN(x[:, 1, :, :], self.devices)
        # feat3 = self.GAPCN(x[:, 2, :, :], self.devices)

        # # 增加一个新的维度
        # feat1 = feat1.unsqueeze(1)
        # feat2 = feat2.unsqueeze(1)
        # feat3 = feat3.unsqueeze(1)
        # # 沿着新的维度进行拼接
        # features = torch.cat((feat1, feat2, feat3), dim=1).permute(0, 1, 3, 2)
        # output, hc, _ = self.lstm1(features) # output: [batchsize, seq, 516, pts, 1]
        # # print(output[0].shape)
        # # conv input: # batch_size * channel * p_num * 1
        # # x = self.conv(output[0][:, enhance_frame, :, :]) # loss:102
        # x = self.conv(hc[0][0]) # [batchsize, 1, pts, 1]
        # # print(x.shape)
        # x = torch.squeeze(x)
        # x = torch.unsqueeze(x, dim=-2)
        # x = x.transpose(2, 1)  # batch_size*p_num*1
        # x = x + rec
# ------------------------------------------- PlanA -------------------------------------------

        
# ------------------------------------------- PlanB -------------------------------------------
        # feat = self.GAPCN(x[:, enhance_frame, :, :], self.devices) # [batchsize, 2048, 576]
        # feat = feat.unsqueeze(-1)
        # _, hc1, _ = self.lstm1(x) # hc1[0][0/1] shape: torch.Size([batchsize, hidden_dim=64, 2048, 1])
        # feat = torch.cat((feat.permute(0, 2, 1, 3), hc1[0][0]), dim=1)
        # x = self.conv(feat)
        # x = torch.squeeze(x)
        # x = torch.unsqueeze(x, dim=-2)
        # x = x.transpose(2, 1)  # batch_size*p_num*1
        # x = x + rec
# ------------------------------------------- PlanB -------------------------------------------


# ------------------------------------------- PlanC -------------------------------------------
        sffe_time_begin = time.time()
        feat1 = self.GAPCN(x[:, 0, :, :], self.devices) # [batchsize, 2048, 576]
        feat2 = self.GAPCN(x[:, 1, :, :], self.devices)
        feat3 = self.GAPCN(x[:, 2, :, :], self.devices)
        sffe_time_end = time.time()
        print('SFFE time:', sffe_time_end - sffe_time_begin, file=sys.stdout)

        # 增加一个新的维度
        feat1 = feat1.unsqueeze(1)
        feat2 = feat2.unsqueeze(1)
        feat3 = feat3.unsqueeze(-1).permute(0, 2, 1, 3)
        # 沿着新的维度进行拼接
        features = torch.cat((feat1, feat2), dim=1).permute(0, 1, 3, 2)
        lstm_time_begin = time.time()
        _, hc, _ = self.lstm1(features)
        lstm_time_end = time.time()
        print('LSTM time:', lstm_time_end - lstm_time_begin, file=sys.stdout)
        st_features = torch.cat((hc[0][0], feat3), dim=1)
        conv_time_begin = time.time()
        x = self.conv(st_features) # [batchsize, 1, pts, 1]
        conv_time_end = time.time()
        print('Conv time:', conv_time_end - conv_time_begin, file=sys.stdout)
        x = torch.squeeze(x)
        x = torch.unsqueeze(x, dim=-2)
        x = x.transpose(2, 1)  # batch_size*p_num*1
        x = x + rec
# ------------------------------------------- PlanC -------------------------------------------

        # feat1 = self.GAPCN1(x[:, 0, :, :1024], self.devices) # [batchsize, 2048, 576]
        # feat2 = self.GAPCN2(x[:, 1, :, :1024], self.devices)
        # # 增加一个新的维度
        # feat1 = feat1.unsqueeze(1)  # [batchsize, 1, 2048, 576]
        # feat2 = feat2.unsqueeze(1)  # [batchsize, 1, 2048, 576]
        # # 沿着新的维度进行拼接
        # feat12 = torch.cat((feat1, feat2), dim=1).permute(0, 1, 3, 2)  # [batchsize, 2, 576, 2048]
        # _, hc1, _ = self.lstm1(feat12) # hc1[0][0/1] shape: torch.Size([batchsize, hidden_dim=64, 2048, 1])
        # output1 = torch.cat((hc1[0][0], hc1[0][1]), dim=3) # [batchsize, hidden_dim=64, 2048, 2]

        # feat3 = self.GAPCN3(x[:, 2, :, :])
        # feat3 = feat3.unsqueeze(1)
        # feat123 = torch.cat((feat1, feat2, feat3), dim=1).permute(0, 1, 3, 2) # [batchsize, 3, 576, 2048]
        # _, hc2, _ = self.lstm2(torch.cat(hc1, feat3), dim=1)
        # x = self.GAPCN2(x)
        # x = self.lstm2(x)

        return x


if __name__ == "__main__":
    a = torch.randn((5, 4, 3, 2))
    print(a)
    print(torch.max(a, dim=2))
    print(torch.max(a, dim=2)[0].size())
    print(len(a))
