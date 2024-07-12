from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from loss import build_target, yolo_loss


class Yolov2_group_norm(nn.Module):
    num_classes = 20
    num_anchors = 5

    def __init__(self, classes=None, weights_file=False):
        super(Yolov2_group_norm, self).__init__()
        self.grads = None
        if classes:
            self.num_classes = len(classes)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.slowpool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        num_groups = 8
        self.gn1 = nn.GroupNorm(num_groups, 16)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(num_groups, 32)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.gn3 = nn.GroupNorm(num_groups, 64)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.gn4 = nn.GroupNorm(num_groups, 128)

        self.conv5 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.gn5 = nn.GroupNorm(num_groups, 256)

        self.conv6 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.gn6 = nn.GroupNorm(num_groups, 512)

        self.conv7 = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.gn7 = nn.GroupNorm(num_groups, 1024)

        self.conv8 = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.gn8 = nn.GroupNorm(num_groups, 1024)

        self.conv9 = nn.Sequential(
            nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1)
        )

    def forward(
        self, x, gt_boxes=None, gt_classes=None, num_boxes=None, training=False
    ):
        """
        x: Variable
        gt_boxes, gt_classes, num_boxes: Tensor
        """
        # exactly 5 pooling layers, like dark net-19:
        x = self.maxpool(self.lrelu(self.gn1(self.conv1(x))))
        x = self.maxpool(self.lrelu(self.gn2(self.conv2(x))))
        x = self.maxpool(self.lrelu(self.gn3(self.conv3(x))))
        x = self.maxpool(self.lrelu(self.gn4(self.conv4(x))))
        x = self.maxpool(self.lrelu(self.gn5(self.conv5(x))))
        x = self.lrelu(self.gn6(self.conv6(x)))
        x = self.lrelu(self.gn7(self.conv7(x)))
        x = self.lrelu(self.gn8(self.conv8(x)))
        out = self.conv9(x)

        # out -- tensor of shape (B, num_anchors * (5 + num_classes), H, W)
        bsize, _, h, w = out.size()

        # 5 + num_class tensor represents (t_x, t_y, t_h, t_w, t_c) and (class1_score, class2_score, ...)
        # reorganize the output tensor to shape (B, H * W * num_anchors, 5 + num_classes)

        # 5 + num_class tensor represents (t_x, t_y, t_h, t_w, t_c) and (class1_score, class2_score, ...)
        # reorganize the output tensor to shape (B, H * W * num_anchors, 5 + num_classes)
        out = (
            out.permute(0, 2, 3, 1)
            .contiguous()
            .view(bsize, h * w * self.num_anchors, 5 + self.num_classes)
        )

        # activate the output tensor
        # `sigmoid` for t_x, t_y, t_c; `exp` for t_h, t_w;
        # `softmax` for (class1_score, class2_score, ...)

        xy_pred = torch.sigmoid(out[:, :, 0:2])
        conf_pred = torch.sigmoid(out[:, :, 4:5])
        hw_pred = torch.exp(out[:, :, 2:4])
        class_score = out[:, :, 5:]
        class_pred = F.softmax(class_score, dim=-1)
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)

        if training:
            output_variable = (delta_pred, conf_pred, class_score)
            output_data = [v.data for v in output_variable]
            gt_data = (gt_boxes, gt_classes, num_boxes)
            target_data = build_target(output_data, gt_data, h, w)

            target_variable = [Variable(v) for v in target_data]
            box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)

            return box_loss, iou_loss, class_loss

        return delta_pred, conf_pred, class_pred

    def calculate_gradients(self, loss):
        self.zero_grad()
        # Calculate gradients
        loss.retain_grad()
        loss.backward(retain_graph=True)

    def manual_weight_update(
        model, learning_rate=0.0001, momentum=0.9, weight_decay=0.0005, dampening=0
    ):
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is None:
                    continue
                d_p = param.grad.data

                if weight_decay != 0:
                    d_p = d_p + weight_decay * param.data
                    # apply learning ate
                d_p = d_p * learning_rate

                if momentum != 0:
                    if not hasattr(param, "momentum_buffer"):
                        param.momentum_buffer = torch.zeros_like(param.data)
                        buffer = param.momentum_buffer
                        buffer = momentum * buffer + d_p
                    else:
                        buffer = param.momentum_buffer
                        buffer = momentum * buffer + (1 - dampening) * d_p
                param.data = param.data - learning_rate * d_p


class Yolov2_minus(nn.Module):
    num_classes = 20
    num_anchors = 5

    def __init__(self, classes=None, weights_file=False):
        super(Yolov2_minus, self).__init__()
        self.grads = None
        if classes:
            self.num_classes = len(classes)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.slowpool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn7 = nn.BatchNorm2d(1024)

        # removing these layers doesn't affect the network:
        # self.conv8 = nn.Conv2d(
        #     in_channels=1024,
        #     out_channels=1024,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     bias=False,
        # )
        # self.bn8 = nn.BatchNorm2d(1024)

        self.conv9 = nn.Sequential(
            nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1)
        )

    def forward(
        self, x, gt_boxes=None, gt_classes=None, num_boxes=None, training=False
    ):
        """
        x: Variable
        gt_boxes, gt_classes, num_boxes: Tensor
        """
        x = self.maxpool(self.lrelu(self.bn1(self.conv1(x))))
        x = self.maxpool(self.lrelu(self.bn2(self.conv2(x))))
        x = self.maxpool(self.lrelu(self.bn3(self.conv3(x))))
        x = self.maxpool(self.lrelu(self.bn4(self.conv4(x))))
        x = self.maxpool(self.lrelu(self.bn5(self.conv5(x))))
        x = self.lrelu(self.bn6(self.conv6(x)))
        x = self.lrelu(self.bn7(self.conv7(x)))
        # comment out them in the forward pass as well:
        # x = self.lrelu(self.bn8(self.conv8(x)))

        out = self.conv9(x)

        # out -- tensor of shape (B, num_anchors * (5 + num_classes), H, W)
        bsize, _, h, w = out.size()
        # print(out.size())

        # 5 + num_class tensor represents (t_x, t_y, t_h, t_w, t_c) and (class1_score, class2_score, ...)
        # reorganize the output tensor to shape (B, H * W * num_anchors, 5 + num_classes)

        # 5 + num_class tensor represents (t_x, t_y, t_h, t_w, t_c) and (class1_score, class2_score, ...)
        # reorganize the output tensor to shape (B, H * W * num_anchors, 5 + num_classes)
        out = (
            out.permute(0, 2, 3, 1)
            .contiguous()
            .view(bsize, h * w * self.num_anchors, 5 + self.num_classes)
        )

        # activate the output tensor
        # `sigmoid` for t_x, t_y, t_c; `exp` for t_h, t_w;
        # `softmax` for (class1_score, class2_score, ...)

        xy_pred = torch.sigmoid(out[:, :, 0:2])
        conf_pred = torch.sigmoid(out[:, :, 4:5])
        hw_pred = torch.exp(out[:, :, 2:4])
        class_score = out[:, :, 5:]
        class_pred = F.softmax(class_score, dim=-1)
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)

        if training:
            output_variable = (delta_pred, conf_pred, class_score)
            output_data = [v.data for v in output_variable]
            gt_data = (gt_boxes, gt_classes, num_boxes)
            target_data = build_target(output_data, gt_data, h, w)

            target_variable = [Variable(v) for v in target_data]
            box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)

            return box_loss, iou_loss, class_loss

        return delta_pred, conf_pred, class_pred

    def calculate_gradients(self, loss):
        self.zero_grad()
        # Calculate gradients
        loss.retain_grad()
        loss.backward(retain_graph=True)

    def manual_weight_update(
        model, learning_rate=0.0001, momentum=0.9, weight_decay=0.0005, dampening=0
    ):
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is None:
                    continue
                d_p = param.grad.data

                if weight_decay != 0:
                    d_p = d_p + weight_decay * param.data
                    # apply learning ate
                d_p = d_p * learning_rate

                if momentum != 0:
                    if not hasattr(param, "momentum_buffer"):
                        param.momentum_buffer = torch.zeros_like(param.data)
                        buffer = param.momentum_buffer
                        buffer = momentum * buffer + d_p
                    else:
                        buffer = param.momentum_buffer
                        buffer = momentum * buffer + (1 - dampening) * d_p
                param.data = param.data - learning_rate * d_p


class Yolov2_plus(nn.Module):
    num_classes = 20
    num_anchors = 5

    def __init__(self, classes=None, weights_file=False):
        super(Yolov2_plus, self).__init__()
        self.grads = None
        if classes:
            self.num_classes = len(classes)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.slowpool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn4 = nn.BatchNorm2d(128)

        """add from here: 2 convolutional layers, along with batch norm. Convolution layers produce outputs of same 
        size:
        source: https://stackoverflow.com/a/77050912"""

        self.conv5 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn6 = nn.BatchNorm2d(128)
        # adding ends here.
        self.conv7 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn8 = nn.BatchNorm2d(512)

        self.conv9 = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn9 = nn.BatchNorm2d(1024)

        self.conv10 = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn10 = nn.BatchNorm2d(1024)

        self.conv11 = nn.Sequential(
            nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1)
        )

    def forward(
        self, x, gt_boxes=None, gt_classes=None, num_boxes=None, training=False
    ):
        """
        x: Variable
        gt_boxes, gt_classes, num_boxes: Tensor
        """

        x = self.maxpool(self.lrelu(self.bn1(self.conv1(x))))
        x = self.maxpool(self.lrelu(self.bn2(self.conv2(x))))
        x = self.maxpool(self.lrelu(self.bn3(self.conv3(x))))
        x = self.maxpool(self.lrelu(self.bn4(self.conv4(x))))
        # add them to the forward pass as well:
        x = self.maxpool(self.lrelu(self.bn5(self.conv5(x))))
        x = self.lrelu(self.bn6(self.conv6(x)))
        x = self.lrelu(self.bn7(self.conv7(x)))
        x = self.lrelu(self.bn8(self.conv8(x)))
        x = self.lrelu(self.bn9(self.conv9(x)))
        x = self.lrelu(self.bn10(self.conv10(x)))
        out = self.conv11(x)

        # out -- tensor of shape (B, num_anchors * (5 + num_classes), H, W)
        bsize, _, h, w = out.size()
        # print(out.size())

        # 5 + num_class tensor represents (t_x, t_y, t_h, t_w, t_c) and (class1_score, class2_score, ...)
        # reorganize the output tensor to shape (B, H * W * num_anchors, 5 + num_classes)
        out = (
            out.permute(0, 2, 3, 1)
            .contiguous()
            .view(bsize, h * w * self.num_anchors, 5 + self.num_classes)
        )

        # activate the output tensor
        # `sigmoid` for t_x, t_y, t_c; `exp` for t_h, t_w;
        # `softmax` for (class1_score, class2_score, ...)

        xy_pred = torch.sigmoid(out[:, :, 0:2])
        conf_pred = torch.sigmoid(out[:, :, 4:5])
        hw_pred = torch.exp(out[:, :, 2:4])
        class_score = out[:, :, 5:]
        class_pred = F.softmax(class_score, dim=-1)
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)

        if training:
            output_variable = (delta_pred, conf_pred, class_score)
            output_data = [v.data for v in output_variable]
            gt_data = (gt_boxes, gt_classes, num_boxes)
            target_data = build_target(output_data, gt_data, h, w)

            target_variable = [Variable(v) for v in target_data]
            box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)

            return box_loss, iou_loss, class_loss

        return delta_pred, conf_pred, class_pred

    def calculate_gradients(self, loss):
        self.zero_grad()
        # Calculate gradients
        loss.retain_grad()
        loss.backward(retain_graph=True)

    def manual_weight_update(
        model, learning_rate=0.0001, momentum=0.9, weight_decay=0.0005, dampening=0
    ):
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is None:
                    continue
                d_p = param.grad.data

                if weight_decay != 0:
                    d_p = d_p + weight_decay * param.data
                    # apply learning ate
                d_p = d_p * learning_rate

                if momentum != 0:
                    if not hasattr(param, "momentum_buffer"):
                        param.momentum_buffer = torch.zeros_like(param.data)
                        buffer = param.momentum_buffer
                        buffer = momentum * buffer + d_p
                    else:
                        buffer = param.momentum_buffer
                        buffer = momentum * buffer + (1 - dampening) * d_p
                param.data = param.data - learning_rate * d_p


class Yolov2(nn.Module):
    num_classes = 20
    num_anchors = 5

    def __init__(self, classes=None, weights_file=False):
        super(Yolov2, self).__init__()
        self.grads = None
        if classes:
            self.num_classes = len(classes)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.slowpool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn7 = nn.BatchNorm2d(1024)

        self.conv8 = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn8 = nn.BatchNorm2d(1024)

        self.conv9 = nn.Sequential(
            nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1)
        )

    def forward(
        self, x, gt_boxes=None, gt_classes=None, num_boxes=None, training=False
    ):
        """
        x: Variable
        gt_boxes, gt_classes, num_boxes: Tensor
        """
        x = self.maxpool(self.lrelu(self.bn1(self.conv1(x))))
        x = self.maxpool(self.lrelu(self.bn2(self.conv2(x))))
        x = self.maxpool(self.lrelu(self.bn3(self.conv3(x))))
        x = self.maxpool(self.lrelu(self.bn4(self.conv4(x))))
        x = self.maxpool(self.lrelu(self.bn5(self.conv5(x))))
        x = self.lrelu(self.bn6(self.conv6(x)))
        x = self.lrelu(self.bn7(self.conv7(x)))
        x = self.lrelu(self.bn8(self.conv8(x)))
        out = self.conv9(x)

        # out -- tensor of shape (B, num_anchors * (5 + num_classes), H, W)
        bsize, _, h, w = out.size()

        # 5 + num_class tensor represents (t_x, t_y, t_h, t_w, t_c) and (class1_score, class2_score, ...)
        # reorganize the output tensor to shape (B, H * W * num_anchors, 5 + num_classes)

        # 5 + num_class tensor represents (t_x, t_y, t_h, t_w, t_c) and (class1_score, class2_score, ...)
        # reorganize the output tensor to shape (B, H * W * num_anchors, 5 + num_classes)
        out = (
            out.permute(0, 2, 3, 1)
            .contiguous()
            .view(bsize, h * w * self.num_anchors, 5 + self.num_classes)
        )

        # activate the output tensor
        # `sigmoid` for t_x, t_y, t_c; `exp` for t_h, t_w;
        # `softmax` for (class1_score, class2_score, ...)

        xy_pred = torch.sigmoid(out[:, :, 0:2])
        conf_pred = torch.sigmoid(out[:, :, 4:5])
        hw_pred = torch.exp(out[:, :, 2:4])
        class_score = out[:, :, 5:]
        class_pred = F.softmax(class_score, dim=-1)
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)

        if training:
            output_variable = (delta_pred, conf_pred, class_score)
            output_data = [v.data for v in output_variable]
            gt_data = (gt_boxes, gt_classes, num_boxes)
            target_data = build_target(output_data, gt_data, h, w)

            target_variable = [Variable(v) for v in target_data]
            box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)

            return box_loss, iou_loss, class_loss

        return delta_pred, conf_pred, class_pred

    def calculate_gradients(self, loss):
        self.zero_grad()
        # Calculate gradients
        loss.retain_grad()
        loss.backward(retain_graph=True)

    def manual_weight_update(
        model, learning_rate=0.0001, momentum=0.9, weight_decay=0.0005, dampening=0
    ):
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is None:
                    continue
                d_p = param.grad.data

                if weight_decay != 0:
                    d_p = d_p + weight_decay * param.data
                    # apply learning ate
                d_p = d_p * learning_rate

                if momentum != 0:
                    if not hasattr(param, "momentum_buffer"):
                        param.momentum_buffer = torch.zeros_like(param.data)
                        buffer = param.momentum_buffer
                        buffer = momentum * buffer + d_p
                    else:
                        buffer = param.momentum_buffer
                        buffer = momentum * buffer + (1 - dampening) * d_p
                param.data = param.data - learning_rate * d_p


if __name__ == "__main__":
    model = Yolov2()
    print(model)
    im = np.random.randn(1, 3, 416, 416)
    im_variable = Variable(torch.from_numpy(im)).float()
    out = model(im_variable)
    delta_pred, conf_pred, class_pred = out
    # some unit tests:
    print("delta_pred size:", delta_pred.size())
    print("conf_pred size:", conf_pred.size())
    print("class_pred size:", class_pred.size())
