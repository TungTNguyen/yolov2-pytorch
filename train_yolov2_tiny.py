from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter

from dataset.factory import get_imdb
from dataset.roidb import RoiDataset, detection_collate
from yolov2_tiny_2 import *
from util.network import adjust_learning_rate
from config import config as cfg
from Test_with_train import test_for_train
from weight_update import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Yolo v2")
    parser.add_argument(
        "--max_epochs",
        dest="max_epochs",
        help="number of epochs to train",
        default=100,
        type=int,
    )
    parser.add_argument("--start_epoch", dest="start_epoch", default=101, type=int)
    parser.add_argument("--batch_size", dest="batch_size", default=8, type=int)
    parser.add_argument("--data_limit", dest="data_limit", default=0, type=int)
    parser.add_argument("--dataset", dest="dataset", default="voc07trainval", type=str)

    parser.add_argument(
        "--nw",
        dest="num_workers",
        help="number of workers to load training data",
        default=8,
        type=int,
    )
    parser.add_argument("--output_dir", dest="output_dir", default="output", type=str)
    parser.add_argument("--use_tfboard", dest="use_tfboard", default=False, type=bool)
    parser.add_argument(
        "--display_interval", dest="display_interval", default=20, type=int
    )
    parser.add_argument("--mGPUs", dest="mGPUs", default=False, type=bool)
    parser.add_argument("--save_interval", dest="save_interval", default=10, type=int)
    parser.add_argument("--cuda", dest="use_cuda", default=True, type=bool)
    parser.add_argument("--resume", dest="resume", default=True, type=bool)
    parser.add_argument(
        "--checkpoint_epoch", dest="checkpoint_epoch", default=160, type=int
    )
    parser.add_argument("--exp_name", dest="exp_name", default="default", type=str)

    args = parser.parse_args()
    return args


def get_dataset(datasetnames):
    names = datasetnames.split("+")
    dataset = RoiDataset(get_imdb(names[0]))
    print("load dataset {}".format(names[0]))
    for name in names[1:]:
        tmp = RoiDataset(get_imdb(name))
        dataset += tmp
        print("load and add dataset {}".format(name))
    return dataset


def train():

    # define the hyper parameters first
    args = parse_args()
    args.lr = cfg.lr
    args.decay_lrs = cfg.decay_lrs
    args.weight_decay = cfg.weight_decay
    args.momentum = cfg.momentum
    args.batch_size = args.batch_size
    args.pretrained_model = os.path.join("output", "yolov2_epoch_100_original.pth")

    print("Called with args:")
    print(args)

    args.imdb_name, args.imdbval_name = get_dataset_names(args.dataset)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load dataset
    print("loading dataset....")
    train_dataset = get_dataset(args.imdb_name)
    if not args.data_limit == 0:
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(0, args.data_limit)
        )
    print("dataset loaded.")

    print("Training Dataset: {}".format(len(train_dataset)))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
        drop_last=True,
    )

    # initialize the model
    print("initialize the model")
    tic = time.time()

    model = Yolov2()

    if args.resume:
        print("load pretrained...")

        pre_trained_checkpoint = torch.load(args.pretrained_model, map_location="cpu")
        model.load_state_dict(pre_trained_checkpoint["model"])
        print(model)
    else:
        print(model)

    toc = time.time()
    print("model loaded: cost time {:.2f}s".format(toc - tic))

    # initialize the optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.use_cuda:
        model.cuda()

    if args.mGPUs:
        model = nn.DataParallel(model)

    # set the model mode to train because we have some layer whose behaviors are different when in training and testing.
    # such as Batch Normalization Layer.
    model.train()

    iters_per_epoch = int(len(train_dataset) / args.batch_size)
    print("Training Iterations: " + str(iters_per_epoch) + "\n")

    min_loss = 100
    max_map = 0
    best_map_score = -1
    best_map_epoch = -1
    best_map_loss = -1

    # Start training
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        loss_temp = 0
        tic = time.time()
        train_data_iter = iter(train_dataloader)
        print()
        for step in tqdm(
            range(iters_per_epoch), desc=f"Epoch {epoch}", total=iters_per_epoch
        ):

            # Randomly select a scale from the specified range
            if cfg.multi_scale and (step + 1) % cfg.scale_step == 0:
                scale_index = np.random.randint(*cfg.scale_range)
                cfg.input_size = cfg.input_sizes[scale_index]

            # Get the next batch of training data
            im_data, boxes, gt_classes, num_obj = next(train_data_iter)

            # Move the data tensors to the GPU
            if args.use_cuda:
                im_data = im_data.cuda()
                boxes = boxes.cuda()
                gt_classes = gt_classes.cuda()
                num_obj = num_obj.cuda()

            # Convert the input data tensor to a PyTorch Variable
            im_data_variable = Variable(im_data)

            # Compute the losses
            box_loss, iou_loss, class_loss = model(
                im_data_variable, boxes, gt_classes, num_obj, training=True
            )

            # Compute the total loss
            loss = box_loss.mean() + iou_loss.mean() + class_loss.mean()

            # Clear gradients
            optimizer.zero_grad()

            # Compute gradients
            loss.retain_grad()
            loss.backward(retain_graph=True)

            optimizer.step()
            loss_temp += loss.item()

        # Show loss after epoch
        toc = time.time()
        loss_temp /= args.display_interval

        iou_loss_v = iou_loss.mean().item()
        box_loss_v = box_loss.mean().item()
        class_loss_v = class_loss.mean().item()

        print(
            f"[epoch %2d][step %4d/%4d] loss: %.4f, lr: %.2e, time cost %.1fs, tiou_loss: %.4f, box_loss: %.4f, class_loss: %.4f"
            % (
                epoch,
                step + 1,
                iters_per_epoch,
                loss_temp,
                optimizer.param_groups[0]["lr"],
                toc - tic,
                iou_loss_v,
                box_loss_v,
                class_loss_v,
            ),
            end=" ",
        )

        loss_temp = 0
        tic = time.time()

        if epoch % args.save_interval == 0:
            save_name = os.path.join(output_dir, "yolov2_epoch_{}.pth".format(epoch))
            torch.save(
                {
                    "model": model.module.state_dict()
                    if args.mGPUs
                    else model.state_dict(),
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "loss": loss.item(),
                },
                save_name,
            )

        # Check minimum loss and save weights for minimum loss
        if loss.item() < min_loss:
            min_loss = loss.item()
            print(
                f"\n\t-->> lower loss at Epoch {epoch}, with loss={round(loss.item(),2)}"
            )

        # Check and save the best mAP
        if (epoch % 50 == 0) and (epoch > 0):
            save_name_temp = os.path.join(output_dir, "temp")
            map = test_for_train(save_name_temp, model, args)
            best_map_score = round((map * 100), 2)
            best_map_epoch = epoch
            print(
                f"\n\t---------------------mAP was at epoch {best_map_epoch}, with mAP={best_map_score}%\n"
            )
    # for validating with the pretrained model only:
    if args.resume:
        save_name_temp = os.path.join(output_dir, "temp")
        map = test_for_train(save_name_temp, model, args)


if __name__ == "__main__":
    train()
