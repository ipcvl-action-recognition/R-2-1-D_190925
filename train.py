import torch
from torch.autograd import Variable
import dataloader
import network
from torch.utils.data import DataLoader
from torch import optim
from dataloader import Le2i_VideoDataset
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import cv2
import visdom       ### python -m visdom.server
import sys
from torch.optim.lr_scheduler import StepLR
from time import sleep
import os
from torchsummary import summary
from Setting import TrainSetting
import utils

import R2Plus1D_model
if __name__ == "__main__":
    # folder_path = './single_data/Video'
    # readfile = './single_data/list'
    # framefile = './single_data/label_Result'
    clip_len = 16
    save_name = ""

    Train_Le2i_Setting = TrainSetting(folder_path='asdf',
                 readfile='asdfasdf',
                 framefile='asdf',
                 clip_len=16,
                 crop_size=112)
    Val_Le2i_Setting = TrainSetting(folder_path='1234',
                                    readfile='zxcv',
                                    framefile='zxcv',
                                    clip_len=16,
                                    crop_size=112)

    resize_resolution = utils.resolution(l=16, h=360, w=640)

    folder_path = './Test_data/Video'
    readfile = './Test_data/list'
    framefile = './Test_data/label_Result'

    ############ Train ###############

    train_dataset = Le2i_VideoDataset(TrainSetting=Train_Le2i_Setting,
                                      resolution=resize_resolution,
                                      is_train=True)
    val_dataset = Le2i_VideoDataset(TrainSetting=Val_Le2i_Setting,
                                    resolution=resize_resolution,
                                    is_train=False)

    batch_size = 16
    lr = 1e-5

    train_dataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                  pin_memory=False)
    # val_dataLoader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
    #                             pin_memory=False)
    # my_model = network.C3DNet().cuda()
    # my_model = network.C3DNet()
    my_model = R2Plus1D_model.R2Plus1DClassifier(num_classes=1, layer_sizes=(2, 2, 2, 2)).cuda()
    #summary(my_model, input_size=(3, 16, 224, 224), device="cpu")

    ############ 웨이트 이어서 학습할 경우 ############
    # checkpoint = torch.load('Weights/Falldown_0816_best_acc_34_0.25_93.57.pt', map_location=lambda storage, loc: storage)
    # my_model.load_state_dict(checkpoint)

    weight = torch.Tensor([0.56, 0.44])
    # criterion = torch.nn.CrossEntropyLoss().cuda()
    # criterion = torch.nn.BCELoss(reduce=False).cuda()
    criterion = torch.nn.BCELoss(reduce=False).cuda()
    # train_params = [{'params': network.get_1x_lr_params(my_model), 'lr': lr},
    #                {'params': network.get_10x_lr_params(my_model), 'lr': lr}]

    # optimizer = optim.Adam(train_params)
    optimizer = optim.Adam(my_model.parameters(), lr=lr, weight_decay=5e-5)
    # optimizer = optim.Adam(my_model.parameters(), lr=lr)

    training_epoch = 10000

    train_loss = []
    val_loss = []
    accuracy_list = []
    vis = visdom.Visdom()
    loss_window = vis.line(X=torch.zeros((1)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='Epoch',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Train Loss']))
    vis = visdom.Visdom()
    acc_window = vis.line(X=torch.zeros((1)).cpu(),
                          Y=torch.zeros((1)).cpu(),
                          opts=dict(xlabel='Epoch',
                                    ylabel='Acc',
                                    title='Validation  Accuracy',
                                    legend=['Val Acc']))
    best_acc = sys.float_info.min

    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    for epoch in range(training_epoch + 1):
        train_epoch_losses = []

        scheduler.step()
        my_model.train()
        print('Epoch:', epoch, 'LR:', scheduler.get_lr())
        isPut = True
        for it, data in enumerate(train_dataLoader):
            x = data[0].cuda()  # buffer #(batch, frame channel h, w)
            y = data[1].cuda()  # label - one_hot
            # print(x.shape)
            optimizer.zero_grad()
            logits = my_model(x).cuda()
            y_pred = torch.sigmoid(logits)

            weight_ = weight[y.float().data.view(-1).long()].view_as(y.float()).cuda()
            loss = criterion(y_pred.squeeze(), y.float())
            loss_class_weighted = loss * weight_
            loss = loss_class_weighted.mean()
            loss.backward()
            optimizer.step()

            if it % 10 == 0:
                print("epoch {0} Iteration [{1}/{2}] Train_Loss : {3:2.4f}".format(epoch, it, len(train_dataLoader),
                                                                                   loss))
                print("y_pred = ", torch.transpose(y_pred, 0, 1))
                print("y = ", y)
                print("logits = ", torch.transpose(logits, 0, 1))
                print()
            train_epoch_losses.append(loss.item())
            if isPut:
                _x = x[0, :]
                _x = _x.permute(1, 0, 2, 3)
                vis.images(_x, win="Img")
                sleep(1)
                # isPut = False
            # del loss
            # del logits

        val_epoch_losses = []

        my_model.eval()

        #####
        # accuracy = 0
        correct = 0
        none, fall_down = 0, 0
        correct_none, correct_falldown = 0, 0
        Val_randIdx = 0

        val_path = './single_data/'
        video_name = 'C026100_0021.mp4'
        readlabel = open('./single_data/C026100_0021.txt', 'r')
        label_list, person_label_box = utils.load_label_file(readlabel, resize_resolution)

        correct_fall_down, correct_none = utils.falldown_counting(label_list)

        clip = []
        count = 0

        buffer = utils.load_video(val_path, video_name, resize_resolution)
        buffer = utils.crop_video_from_label(buffer, person_label_box, crop_size=112)
        buffer = buffer.transpose((1,2,3,0)) # 3, l, h, w --> l, h, w, 3
        buffer = buffer.astype(np.float32)/255.0
        frame_length = buffer.shape[0]
        for idx, frame in enumerate(buffer):
            clip.append(frame)
            if len(clip) == clip_len: # l, h, w, 3
                input = np.transpose(clip, (3, 0, 1, 2))
                input = torch.from_numpy(input).cuda()

                with torch.no_grad():
                    output = my_model(input)
                output = torch.sigmoid(output).cpu().squeeze()
                y_pred_index = torch.round(output).int()
                if y_pred_index == 0:
                    none += 1
                else:
                    fall_down += 1
                lable_temp = label_list[idx - clip_len + 1: idx+clip_len]
                if lable_temp.count('1') >= int(len(lable_temp) / 2):
                    label = np.array(1)
                else:
                    label = np.array(0)
                y = torch.from_numpy(label).int()
                correct += (y_pred_index == y).sum().item()
                clip.pop(0)

        accuracy = 100 * (correct / frame_length - clip_len + 1)

        print("[None/C_None] : [{0}/{1}], [Fall_Down/C_Falldown] : [{2}/{3}], "
              "[Correct/Total] : [{4}/{5}], [Accuracy] : {6:2.2f}".
              format(none, correct_none, fall_down, correct_falldown, correct, frame_length - clip_len + 1, accuracy))

        mean_epoch_train_loss = np.mean(train_epoch_losses)

        vis.line(X=np.column_stack(np.array([epoch])), Y=np.column_stack(np.array([mean_epoch_train_loss])),
                 win=loss_window, update='append')

        vis.line(X=np.array([epoch]), Y=np.array([accuracy]), win=acc_window, update='append')

        print("epoch {0} Train_mean_Loss : {1:2.4f}".format(epoch, mean_epoch_train_loss))
        print("val_accuracy : ", accuracy)
        print()

        if best_acc < accuracy:
            print("Saved New Weight {0:2.2f} to {1:2.2f} acc".format(best_acc, accuracy))
            best_acc = accuracy
            utils.save_model(my_model, save_name=save_name, acc=accuracy, epoch=epoch)

        if epoch % 100 == 0:
            utils.save_model(my_model, save_name=save_name, acc=accuracy, epoch=epoch)

        if epoch % 50 == 0:
            utils.save_model(my_model, save_name=save_name, acc=accuracy, epoch=epoch)
        else:
            print("There are no improve than {0:2.2f} acc".format(best_acc))