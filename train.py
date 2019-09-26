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
import R2Plus1D_model
if __name__ == "__main__":
    # folder_path = './single_data/Video'
    # readfile = './single_data/list'
    # framefile = './single_data/label_Result'


    folder_path = './Test_data/Video'
    readfile = './Test_data/list'
    framefile = './Test_data/label_Result'


    Val_folder_path = folder_path + '/Val'
    Val_readfile = readfile + '/vallist.txt'
    Val_framefile = framefile + '/Val'

    folder_list = os.listdir(Val_folder_path)
    frame_folder_list = os.listdir(Val_framefile)
    clip_len = 16
    crop_size_width = 112
    crop_size_height = 112
    resize_width = 320
    resize_height = 240

    ############ Train ###############
    train_dataset = Le2i_VideoDataset(folder_path=folder_path + '/Train', readfile=readfile + '/trainlist.txt',
                                      framefile=framefile + '/Train', clip_len=clip_len,
                                      crop_size_width=crop_size_width, crop_size_height=crop_size_height, is_train=True)
    # train_dataset = Le2i_VideoDataset(folder_path=folder_path, readfile=readfile + '/trainlist.txt',
    #                                   framefile=framefile, clip_len=clip_len, crop_size_width=crop_size_width, crop_size_height=crop_size_height, is_train=True)
    # val_dataset = Le2i_VideoDataset(folder_path=folder_path + '/Val', readfile=readfile + '/vallist.txt',
    #                                 framefile=framefile + '/Val', clip_len=clip_len, crop_size_width=crop_size_width, crop_size_height=crop_size_height, is_train=False)
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
            print()
            # print("1 :", y_pred)
            # print("2 :", y_pred.squeeze())
            # print("3 :", y.float())
            weight_ = weight[y.float().data.view(-1).long()].view_as(y.float()).cuda()
            # print("4 : ", weight_)
            loss = criterion(y_pred.squeeze(), y.float())
            # print("5 : ", loss)
            loss_class_weighted = loss * weight_
            # print("6 : ", loss_class_weighted)
            loss = loss_class_weighted.mean()
            # print("7 : ", loss)
            # loss_class_weighted = loss * weight_.double().mean()
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
        # Val_randIdx = int(np.random.uniform(0, len(folder_list)))
        #
        video = (Val_folder_path + '/' + folder_list[Val_randIdx])
        readlabel = open((Val_framefile + '/' + frame_folder_list[Val_randIdx]), 'r')
        # video = './single_data/Coffee_room_01 (02).mp4'
        # readlabel = open('./single_data/Coffee_room_01 (02).txt', 'r')
        label_list = []
        label_person_box = []
        for i, line in enumerate(readlabel):
            # if startFrame <= i < (startFrame + self.clip_len):
            line = line.strip()
            line = line.split(' ')
            label_list.append(line[1])
            label_person_box.append((line[2], line[3], line[4], line[5]))
            if int(line[1]) == 1:
                correct_falldown += 1
            else:
                correct_none += 1

        cap = cv2.VideoCapture(video)
        retaining = True

        clip = []
        count = 0
        frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while retaining:
            retaining, frame = cap.read()
            if not retaining and frame is None:
                continue
            frame = cv2.resize(frame, (resize_width, resize_height))
            # center_x = int(int(label_person_box[count][0])) + int(
            #     (int(label_person_box[count][2]) / 2))  # x + w/2 중점
            # center_y = int(int(label_person_box[count][1])) + int(
            #     (int(label_person_box[count][3]) / 2))  # y + h/2
            # print("center x_y", center_x, center_y)
            person_x = round((float(label_person_box[count][0]) / float(frame_width)) * resize_width)
            person_y = round((float(label_person_box[count][1]) / float(frame_height)) * resize_height)
            person_w = round((float(label_person_box[count][2]) / float(frame_width)) * resize_width)
            person_h = round((float(label_person_box[count][3]) / float(frame_height)) * resize_height)

            center_x = int(person_x + int(person_w / 2))
            center_y = int(person_y + int(person_h / 2))

            LeftTopx = center_x - crop_size_width
            LeftTopy = center_y - crop_size_height
            RightBottomx = center_x + crop_size_width
            RightBottomy = center_y + crop_size_height

            if (LeftTopx < 0):
                LeftTopx = 0
                RightBottomx = (crop_size_width * 2)
            if (LeftTopy < 0):
                LeftTopy = 0
                RightBottomy = (crop_size_height * 2)

            if (RightBottomx > frame.shape[1] - 1):
                RightBottomx = frame.shape[1] - 1
                LeftTopx = (frame.shape[1] - 1) - (crop_size_width * 2)

            if (RightBottomy > frame.shape[0] - 1):
                RightBottomy = frame.shape[0] - 1
                LeftTopy = (frame.shape[0] - 1) - (crop_size_height * 2)
            frame_temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tmp_ = frame_temp[LeftTopy:RightBottomy, LeftTopx:RightBottomx, :]

            tmp = tmp_.astype(np.float32)
            tmp = tmp / 255.0
            clip.append(tmp)
            if len(clip) == clip_len:
                inputs = np.array(clip).astype(np.float32)  # input_shape =  (16, 112, 112, 3)
                inputs = np.expand_dims(inputs, axis=0)  # input_shape =  (1, 16, 112, 112, 3)
                inputs = np.transpose(inputs, (0, 4, 1, 2, 3))  # input_shape =  (1, 3, 16, 112, 112)
                inputs = torch.from_numpy(inputs)
                inputs = torch.autograd.Variable(inputs, requires_grad=False).cuda()

                # startTime = time.time()
                with torch.no_grad():
                    #outputs = my_model.forward(inputs)
                    outputs = my_model(inputs)
                # endTime = time.time() - startTime
                # print(endTime)
                # print("outputs = ", outputs)                               # outputs =  tensor([[-0.0424]], device='cuda:0')
                # outputs = np.squeeze(torch.sigmoid(outputs).cpu().numpy())
                outputs = torch.sigmoid(outputs).cpu().squeeze()
                # print("outputs = ", outputs)                                 # outputs =  0.53555995
                # print("y_pred_index : ", outputs)
                y_pred_index = torch.round(outputs).int()
                if y_pred_index == 0:
                    none += 1
                else:
                    fall_down += 1

                label = label_list[count: count + clip_len]
                label = np.array(int(label[-1]))
                y = torch.from_numpy(label).int()

                # if isPut:
                #     _x2 = inputs[0, :]
                #     _x2 = _x2.permute(1, 0, 2, 3)
                #     vis.images(_x2, win="Img2")
                #     sleep(1)

                # print("y_pred_index : ", y_pred_index)
                # print("y : ", y)
                correct += (y_pred_index == y).sum().item()
                # cv2.imshow('tmp_', tmp_)
                # cv2.waitKey(1)
                clip.pop(0)
            count += 1
        accuracy = 100 * (correct / frame_length)

        print("[None/C_None] : [{0}/{1}], [Fall_Down/C_Falldown] : [{2}/{3}], "
              "[Correct/Total] : [{4}/{5}], [Accuracy] : {6:2.2f}".
              format(none, correct_none, fall_down, correct_falldown, correct, frame_length - clip_len + 1, accuracy))
        # if accuracy < 40:
        #     print("Low accuracy !!! accuracy: ", accuracy , "filename : ", folder_list[Val_randIdx])
        #     torch.save(my_model.state_dict(),
        #                'Weights/low_accuracy{0}_{1:2.2f}_{2}.pt'.format(epoch + 1, accuracy, folder_list[Val_randIdx]))
        cap.release()
        # cv2.destroyAllWindows()


        mean_epoch_train_loss = np.mean(train_epoch_losses)

        vis.line(X=np.column_stack(np.array([epoch])), Y=np.column_stack(np.array([mean_epoch_train_loss])),
                 win=loss_window, update='append')


        vis.line(X=np.array([epoch]), Y=np.array([accuracy]), win=acc_window, update='append')

        print("epoch {0} Train_mean_Loss : {1:2.4f}".format(epoch, mean_epoch_train_loss))
        print("val_accuracy : ", accuracy)
        print()
        print()
        date = "0918_21D_224x224x16_ChannelReduce"
        if best_acc < accuracy:
            print("Saved New Weight {0:2.2f} to {1:2.2f} acc".format(best_acc, accuracy))
            best_acc = accuracy
            torch.save(my_model.state_dict(),
                       'Weights/Falldown_{0}_best_acc_{1}_{2:2.2f}_{3:2.2f}.pt'.format(date, epoch + 1,
                                                                                                mean_epoch_train_loss,
                                                                                                best_acc))
            print()
            print()

        if epoch % 100 == 0:
            torch.save(my_model.state_dict(),
                       'Weights/Falldown_{0}_{1}_{2:2.2f}_{3:2.2f}.pt'.format(date, epoch + 1,
                                                                                                mean_epoch_train_loss,
                                                                                                best_acc))

        if epoch % 50 == 0:
            torch.save(my_model.state_dict(),
                       'Weights/Falldown_{0}_{1}_{2:2.2f}_{3:2.2f}.pt'.format(date, epoch + 1,
                                                                                                mean_epoch_train_loss,
                                                                                                best_acc))

        else:
            print("There are no improve than {0:2.2f} acc".format(best_acc))