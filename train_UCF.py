import torch
from torch.autograd import Variable
import dataloader
import network
from torch.utils.data import DataLoader
from torch import optim
from dataloader import Le2i_VideoDataset
from dataloader import UCF101_Dataset
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import cv2
# import visdom
import sys
from networks import R2Plus1D_model
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

if __name__ == "__main__":
    UCF_path = 'D:/DATASET/UCF-101'
    readfile = 'D:/DATASET/UCF-101_recognition'
    batch_size = 2
    lr = 1e-2
    train_dataset = UCF101_Dataset(folder_path=UCF_path, readfile=readfile+'/trainlist01.txt')
    val_dataset = UCF101_Dataset(folder_path=UCF_path, readfile=readfile+'/testlist01.txt')

    train_dataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                  pin_memory=False)
    val_dataLoader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                pin_memory=False)
    # my_model = network.C3DNet(pretrained=True).cuda()
    my_model = R2Plus1D_model.R2Plus1DClassifier(101, (2, 2, 2, 2), pretrained=False).cuda()
    train_params = [{'params': R2Plus1D_model.get_1x_lr_params(my_model), 'lr': lr},
                    {'params': R2Plus1D_model.get_10x_lr_params(my_model), 'lr': lr * 10}]
    criterion = torch.nn.CrossEntropyLoss().cuda()
    # criterion = torch.nn.BCELoss().cuda()

    # train_params = [{'params': network.get_1x_lr_params(my_model), 'lr': lr},
    #                {'params': network.get_10x_lr_params(my_model), 'lr': lr}]

    # optimizer = optim.Adam(train_params)
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(my_model.parameters(), lr=lr, weight_decay=5e-5)


    training_epoch = 1000
    legend_chekced = True
    train_loss = []
    val_loss = []
    accuracy_list = []
    '''
    vis = visdom.Visdom()
    
    loss_window = vis.line(X=torch.zeros((1, 2)).cpu(),
                           Y=torch.zeros((1, 2)).cpu(),
                           opts=dict(xlabel='Epoch',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Train Loss', 'Val Loss']))

    acc_window = vis.line(X=torch.zeros((1)).cpu(),
                          Y=torch.zeros((1)).cpu(),
                          opts=dict(xlabel='Epoch',
                                    ylabel='Acc',
                                    title='Validation  Accuracy',
                                    legend=['Val Acc']))
    '''
    best_acc = sys.float_info.min

    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    for epoch in range(training_epoch + 1):
        x_line = [i + 1 for i in range(epoch + 1)]
        train_epoch_losses = []
        accuracy = 0
        correct = 0

        scheduler.step()
        my_model.train()
        print('Epoch:', epoch, 'LR:', scheduler.get_lr())
        isPut = True

        for it, data in enumerate(train_dataLoader):
            x = data[0].cuda()  # buffer
            y = data[1].cuda()  # label - one_hot

            optimizer.zero_grad()

            logits = my_model(x).cuda()
            # print(logits.shape)
            softmax = nn.Softmax()
            y_pred = softmax(logits)
            print(y_pred.shape, y.shape)
            loss = criterion(logits, y.long())

            loss.backward()
            optimizer.step()

            if it % 1000 == 0:
                print("epoch {0} Iteration [{1}/{2}] Train_Loss : {3:2.4f}".format(epoch, it, len(train_dataLoader),
                                                                                   loss))
                print("y = ", y)
                print("y_pred =  ", torch.argmax(logits, dim=1))
                print()
            train_epoch_losses.append(loss.item())

            if isPut:
                _x = x[0, :]
                _x = _x.permute(1, 0, 2, 3)
                # vis.images(_x, win="Img")
                isPut = False
            del loss
            del logits

        val_epoch_losses = []
        my_model.eval()
        total_len = 0
        for it, data in enumerate(val_dataLoader):
            with torch.no_grad():
                x = data[0].cuda()
                y = data[1].cuda()

                logits = my_model(x).cuda()
                softmax = nn.Softmax(dim=1)
                y_pred = softmax(logits)
                loss = criterion(y_pred, y.long())
                y_pred_index = torch.argmax(y_pred, dim=1).int()
                y = y.int()
                correct += (y_pred_index == y).sum().item()
                val_epoch_losses.append(loss.item())

        mean_epoch_train_loss = np.mean(train_epoch_losses)
        mean_epoch_val_loss = np.mean(val_epoch_losses)
        train_loss.append(mean_epoch_train_loss)
        val_loss.append(mean_epoch_val_loss)
        # vis.line(X=np.column_stack((np.array([epoch]), np.array([epoch]))),
        #          Y=np.column_stack((np.array([mean_epoch_train_loss]), np.array([mean_epoch_val_loss]))),
        #          win=loss_window, update='append')

        # acc = 100 * (correct / total_len)
        acc = 100 * (correct / val_dataset.__len__())
        accuracy_list.append(acc)
        # vis.line(X=np.array([epoch]), Y=np.array([acc]), win=acc_window, update='append')

        print("epoch {0} Train_mean_Loss : {1:2.4f}  val_mean_Loss : {2:2.4f}".format(epoch, mean_epoch_train_loss,
                                                                                      mean_epoch_val_loss))
        print("val_accuracy : ", acc)
        print()
        print()
        plt.subplot(211)
        plt.plot(x_line, train_loss, 'r-', label='train')
        plt.plot(x_line, val_loss, 'b-', label='val')
        if legend_chekced:
            plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('R(2+1)D Network')
        plt.subplot(212)
        plt.plot(x_line, accuracy_list, 'g-', label='accuracy')
        plt.ylabel('Accuracy')
        if legend_chekced:
            legend_chekced=False
            plt.legend()
        plt.savefig('R(2+1)D Network_test.png', dpi=300)

        if best_acc < acc or epoch+1 % 10 == 0:
            print("Saved New Weight {0:2.2f} to {1:2.2f} acc".format(best_acc, acc))
            best_acc = acc
            torch.save(my_model.state_dict(),
                       'Weights/R(2+1D)_{0}_{1:2.2f}_{2:2.2f}_{3:2.2f}.pt'.format(epoch + 1, mean_epoch_train_loss,
                                                                                   mean_epoch_val_loss, best_acc))
        else:
            print("There are no improve than {0:2.2f} acc".format(best_acc))