"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


configs_dict = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

class softmax_cross_entropy_loss_F(torch.autograd.Function):
  """ my version of masked tf.nn.softmax_cross_entropy_with_logits
  """
  @staticmethod
  def forward(ctx, input, target):
    if not target.is_same_size(input):
      raise ValueError(
        "Target size ({}) must be the same as input size ({})".format(
          target.size(), input.size()
        )
      )
    ctx.save_for_backward(input, target)
    input = F.softmax(input, dim=1)
    loss = -target * torch.log(input)
    loss = torch.sum(loss, 1)
    loss = torch.unsqueeze(loss, 1)
    # return torch.mean(loss)
    return torch.sum(loss)
  @staticmethod
  def backward(ctx, grad_output):
    """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
    input, target = ctx.saved_tensors
    return  F.softmax(input, dim=1)-target, None
class softmax_cross_entropy_loss(nn.Module):
  """ my version of masked tf.nn.softmax_cross_entropy_with_logits
  """
  def __init__(self):
    super(softmax_cross_entropy_loss, self).__init__()
  def forward(self, input, target):
    if not target.is_same_size(input):
      raise ValueError(
        "Target size ({}) must be the same as input size ({})".format(
          target.size(), input.size()
        )
      )
    return softmax_cross_entropy_loss_F.apply(input, target)
        
class VggLoc(nn.Module):
    def __init__(self, num_classes=1000, **kwargs):
        super(VggLoc, self).__init__()
        self.features = make_layers(configs_dict['D'])
        self.cls_loss_dgl = softmax_cross_entropy_loss()
        self.layers_list = {}
        self.grads_list = {}
        self.isgrad_dgl = True
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def save_grad(self, name):
        def hook(grad):
            self.grads_list[name] = grad
        return hook
        
    def forward(self, x):
        x = self.features[0](x)
        x = self.features[1](x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv1_1'))
            self.layers_list['conv1_1'] = x
        x = self.features[2](x)
        x = self.features[3](x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv1_2'))
            self.layers_list['conv1_2'] = x
        x = self.features[4](x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('pool1'))
            self.layers_list['pool1'] = x
        
        x = self.features[5](x)
        x = self.features[6](x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv2_1'))
            self.layers_list['conv2_1'] = x
        x = self.features[7](x)
        x = self.features[8](x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv2_2'))
            self.layers_list['conv2_2'] = x
        x = self.features[9](x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('pool2'))
            self.layers_list['pool2'] = x
        
        x = self.features[10](x)
        x = self.features[11](x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv3_1'))
            self.layers_list['conv3_1'] = x
        x = self.features[12](x)
        x = self.features[13](x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv3_2'))
            self.layers_list['conv3_2'] = x
        x = self.features[14](x)
        x = self.features[15](x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv3_3'))
            self.layers_list['conv3_3'] = x
            
        x = self.features[16](x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('pool3'))
            self.layers_list['pool3'] = x
            
        x = self.features[17](x)
        x = self.features[18](x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv4_1'))
            self.layers_list['conv4_1'] = x
            
        x = self.features[19](x)
        x = self.features[20](x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv4_2'))
            self.layers_list['conv4_2'] = x
            
        x = self.features[21](x)
        x = self.features[22](x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv4_3'))
            self.layers_list['conv4_3'] = x
        
        x = self.features[23](x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('pool4'))
            self.layers_list['pool4'] = x
            
        x = self.features[24](x)
        x = self.features[25](x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv5_1'))
            self.layers_list['conv5_1'] = x
        x = self.features[26](x)
        x = self.features[27](x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv5_2'))
            self.layers_list['conv5_2'] = x
        x = self.features[28](x)
        x = self.features[29](x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('conv5_3'))
            self.layers_list['conv5_3'] = x
        x = self.features[30](x)
        if self.isgrad_dgl:
            x.register_hook(self.save_grad('pool5'))
            self.layers_list['pool5'] = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[3](x)
        x = self.classifier[4](x)
        logits = self.classifier[6](x)
        return logits
            
    
    def get_Layercam(self, x, target_layer_name):
        logits = self.forward(x)
        with torch.enable_grad():
            prob = F.softmax(logits, dim=1)
            prob_top1 = prob.topk(5, dim=1)[1].long().detach()[:,0]
            batch_size = x.shape[0]
            sn = self.layers_list[target_layer_name]
            lc_top1 = logits[range(batch_size), prob_top1].sum() 
            self.zero_grad()
            lc_top1.backward(retain_graph=True)
            g_lc2_wrt_sn_top1 = self.grads_list[target_layer_name].clone()
            Mc_top1 = F.relu( (F.relu(g_lc2_wrt_sn_top1)*sn).sum(dim=1, keepdim=False)  )
        return Mc_top1.detach().cpu().numpy().astype(np.float)

    def get_SGLG1(self, x, target_layer_name):
        logits = self.forward(x)
        with torch.enable_grad():
            prob = F.softmax(logits, dim=1)
            prob_top1 = prob.topk(5, dim=1)[1].long().detach()[:,0]
            batch_size = x.shape[0]
            sn = self.layers_list[target_layer_name]
            sn_norm = sn / sn.view(batch_size, -1).norm(dim=1).view(batch_size, 1, 1, 1)
            lc_top1 = logits[range(batch_size), prob_top1].sum() 
            self.zero_grad()
            lc_top1.backward(retain_graph=True)
            g_lc2_wrt_sn_top1 = self.grads_list[target_layer_name].clone()
            g_lc2_wrt_sn_norm_top1 = g_lc2_wrt_sn_top1 / g_lc2_wrt_sn_top1.view(batch_size, -1).norm(dim=1).view(batch_size, 1, 1, 1)
            Mc_top1 = ((sn_norm + g_lc2_wrt_sn_norm_top1 )*g_lc2_wrt_sn_norm_top1).sum(dim=1, keepdim=False)
        return Mc_top1.detach().cpu().numpy().astype(np.float)

    def get_SGLG2(self, x, target_layer_name):
        logits = self.forward(x)
        batch_size = x.shape[0]
        with torch.enable_grad():
            prob = F.softmax(logits, dim=1)
            prob_top1 = prob.topk(5, dim=1)[1].long().detach()[:,0]
            sn = self.layers_list[target_layer_name]
            sn_norm = sn / sn.view(batch_size, -1).norm(dim=1).view(batch_size, 1, 1, 1)
            self.zero_grad()
            weighted_one_hot = torch.zeros([batch_size, 1000]).cuda().float()
            weighted_one_hot[range(batch_size), prob_top1] = 10
            loss = self.cls_loss_dgl(logits,  weighted_one_hot)
            loss.backward(retain_graph=True)
            cls_loss_grad = self.grads_list[target_layer_name].clone()
            grad_norm = cls_loss_grad / cls_loss_grad.view(batch_size, -1).norm(dim=1).view(batch_size, 1, 1, 1)
            sn_norm = sn / sn.view(batch_size, -1).norm(dim=1).view(batch_size, 1, 1, 1)
            An = sn_norm - grad_norm
            Mc_top1 = (-1* grad_norm * An).sum(dim=1, keepdim=False)
        return Mc_top1.detach().cpu().numpy().astype(np.float)

    def get_DGL(self, x, target_layer_name):
        logits = self.forward(x)
        batch_size = x.shape[0]
        with torch.enable_grad():
            prob = F.softmax(logits, dim=1)
            prob_top1 = prob.topk(5, dim=1)[1].long().detach()[:,0]
            sn = self.layers_list[target_layer_name]
            lc_top1 = logits[range(batch_size), prob_top1].sum() 
            sn_norm = sn / sn.view(batch_size, -1).norm(dim=1).view(batch_size, 1, 1, 1)
            
            self.zero_grad()
            lc_top1.backward(retain_graph=True)
            g_lc2_wrt_sn_top1 = self.grads_list[target_layer_name].clone()
            g_lc2_wrt_sn_norm_top1 = g_lc2_wrt_sn_top1 / g_lc2_wrt_sn_top1.view(batch_size, -1).norm(dim=1).view(batch_size, 1, 1, 1)
            
            self.zero_grad()
            weighted_one_hot_top1 = torch.zeros([batch_size, 1000]).cuda().float()
            weighted_one_hot_top1[range(batch_size), prob_top1] = 10
            loss_top1 = self.cls_loss_dgl(logits,  weighted_one_hot_top1)
            loss_top1.backward(retain_graph=True)
            cls_loss_grad_top1 = self.grads_list[target_layer_name].clone()
            grad_norm_top1 = cls_loss_grad_top1 / cls_loss_grad_top1.view(batch_size, -1).norm(dim=1).view(batch_size, 1, 1, 1)
            
            An_top1 = sn_norm - grad_norm_top1
            Mc_top1 = (g_lc2_wrt_sn_norm_top1 * An_top1).sum(dim=1, keepdim=False)
        return Mc_top1.detach().cpu().numpy().astype(np.float)

def make_layers(cfg, **kwargs):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

