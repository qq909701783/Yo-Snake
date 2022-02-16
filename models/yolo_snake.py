# -*- coding: utf-8 -*-
from addict import Dict
from torch import nn
import math
import yaml
import torch
from models.modules.common import Conv
from models.backbone import build_backbone
from models.neck import build_neck
from models.head import build_head
from utils.torch_utils import initialize_weights, fuse_conv_and_bn, model_info
from models.evolve import Evolution
from datasets.dataset import Dataset
from datasets.collate_batch import snake_collator
import torch.nn.functional as F
from utils.general import non_max_suppression
from utils.snake import snake_config


class Model(nn.Module):
    def __init__(self, model_config):
        """
        :param model_config:
        """

        super(Model, self).__init__()
        if type(model_config) is str:
            model_config = yaml.load(open(model_config, 'r'))
        model_config = Dict(model_config)
        backbone_type = model_config.backbone.pop('type')
        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        backbone_out = self.backbone.out_shape

        self.fpn = build_neck('FPN', **backbone_out)
        fpn_out = self.fpn.out_shape

        fpn_out['version'] = model_config.backbone.version
        self.pan = build_neck('PAN', **fpn_out)

        pan_out = self.pan.out_shape
        model_config.head['ch'] = pan_out
        self.detection = build_head('YOLOHead', **model_config.head)
        self.stride = self.detection.stride
        self._initialize_biases()

        initialize_weights(self)

        self.Conv_ = Conv(128,64)
        self.gcn = Evolution()


    def _initialize_biases(self, cf=None):
        # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.detection  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for module in [self.backbone, self.fpn, self.pan, self.detection]:
            for m in module.modules():
                if type(m) is Conv and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


    def use_gt_detection(self, output, batch):
        _, _, height, width = output['ct_hm'].size()
        ct_01 = batch['ct_01'].byte().bool()

        ct_ind = batch['ct_ind'][ct_01]
        xs, ys = ct_ind % width, ct_ind // width
        xs, ys = xs[:, None].float(), ys[:, None].float()
        ct = torch.cat([xs, ys], dim=1)

        wh = batch['wh'][ct_01]
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=1)
        score = torch.ones([len(bboxes)]).to(bboxes)[:, None]
        ct_cls = batch['ct_cls'][ct_01].float()[:, None]
        detection = torch.cat([bboxes, score, ct_cls], dim=1)

        output['ct'] = ct[None]
        output['detection'] = detection[None]

        return output

    def forward(self, x,batch=None):
        use_gt_det = True
        out = self.backbone(x)
        out = self.fpn(out)
        out = self.pan(out)
        cnn_feature = out[0]
        cnn_feature = F.interpolate(cnn_feature, scale_factor=2,mode='bilinear',align_corners=False)
        cnn_feature = self.Conv_(cnn_feature)
        y = self.detection(list(out))

        ###
        output = {}
        if self.training:
            ct_hm = batch['ct_hm']
            output.update({'ct_hm': ct_hm})
            output = self.gcn(output, cnn_feature, batch)
        # elif use_gt_det:
        #     ct_hm = batch['ct_hm']
        #     output.update({'ct_hm': ct_hm})
        #     self.use_gt_detection(output, batch)
        #     print(output)
        else:
            pred = y[0]
            pred = non_max_suppression(pred, 0.2, 0.3, classes=None,agnostic=False)
            pred = torch.cat(tuple(pred), 0)
            ct = torch.cat([(pred[...,0]+pred[...,2])/2,(pred[...,1]+pred[...,3])/2]).unsqueeze(0)
            ct = ct / snake_config.down_ratio
            pred[...,0:4] /= snake_config.down_ratio
            output['ct'] = ct[None]
            output['detection'] = pred[None]
            output = self.gcn(output, cnn_feature, batch)
        ###

        return y,output



def to_cuda(batch,device):
    for k in batch:
        if k == 'meta' or k=='path':
            continue
        if isinstance(batch[k], tuple):
            batch[k] = [b.to(device) for b in batch[k]]
        else:
            batch[k] = batch[k].to(device)

    return batch