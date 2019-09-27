import torch
import cv2
import numpy as np
import os
import visdom
from time import sleep
#####  Video Function  #####

class resolution:
    def __init__(self, l, h, w):
        self.length = l
        self.height = h
        self.width = w
        
def load_video(path, video_name, resolution):
    cap = cv2.VideoCapture(path+video_name)
    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buffer = np.empty((frame_length, frame_height, frame_width, 3), np.dtype('float32'))
    count = 0
    while count < frame_length and cap.isOpened():
        ret, frame = cap.read()  # frame 읽어오면 ret = true 실패하면 ret = false
        if ret:
            if (frame_height != resolution.h) or (frame_width != resolution.w):
                frame = cv2.resize(frame, (resolution.w, resolution.h))  # frame_resize w, h
            buffer[count] = frame
            count += 1
        else:
            break

    buffer = buffer.transpose((3, 0, 1, 2))
    return buffer

def pre_processing(buffer, resolution, crop_size):
    buffer = buffer/255
    # buffer = crop(buffer, clip_len=16, crop_size=crop_size)
    return buffer

def crop(buffer, clip_len, crop_size):
    time_index = np.random.randint(buffer.shape[1] - clip_len) # 196-8
    # print("time_index = ", time_index ) # 186
    height_index = np.random.randint(buffer.shape[2] - crop_size)
    # print("height_index = ", height_index) # 3
    width_index = np.random.randint(buffer.shape[3] - crop_size)
    # print("width_index = ", width_index) # 5

    buffer = buffer[:, time_index:time_index + clip_len,
             height_index:height_index + crop_size,
             width_index:width_index + crop_size]
    return buffer

def model_test(resolution, img_show=False):
    path = ""
    video_name = ""
    buffer = load_video(path, video_name, resolution)
    
    if img_show:
        cv2.imshow("frame", buffer)
        cv2.waitKey(1)

#####  Label Function  #####
def load_label_file(label_file, resolution):
    label_list = []
    person_label_box = []
    readlabel = open(label_file, 'r')
    for i, line in enumerate(readlabel):
        line = line.strip()
        line = line.split()
        label_list.append(line[1])
        person_label_box.append((float(line[2]) / resolution.width,
                                float(line[3]) / resolution.height,
                                float(line[4]) / resolution.width,
                                float(line[5]) / resolution.height))
        # person_label_box.append((line[2], line[3], line[4], line[5]))
    
    return label_list, person_label_box

def falldown_counting(label_list):
    fall_down = 0
    none = 0
    for one in label_list:
        if one == 1:
            fall_down += 1
        else:
            none += 1
    return fall_down, none

def crop_video_from_label(video, person_label_box, crop_size):
    video = video.transpose((1, 2, 3, 0))  # 3, l, h, w --> l ,h, w, 3
    l, h, w, c = video.shape
    output = np.empty((l, crop_size, crop_size, 3), np.dtype('float32'))
    for idx, img in enumerate(video):

        person_x = round((float(person_label_box[idx][0]) * w))
        person_y = round((float(person_label_box[idx][1]) / h))
        person_w = round((float(person_label_box[idx][2]) / w))
        person_h = round((float(person_label_box[idx][3]) / h))
        
        center_x = int(person_x + int(person_w / 2)) * w
        center_y = int(person_y + int(person_h / 2)) * h
        
        LeftTopx = center_x - crop_size
        LeftTopy = center_y - crop_size
        RightBottomx = center_x + crop_size
        RightBottomy = center_y + crop_size

        if (LeftTopx < 0):
            LeftTopx = 0
            RightBottomx = (crop_size * 2)
        if (LeftTopy < 0):
            LeftTopy = 0
            RightBottomy = (crop_size * 2)

        if (RightBottomx > w - 1):
            RightBottomx = w - 1
            LeftTopx = (w - 1) - (crop_size * 2)

        if (RightBottomy > h - 1):
            RightBottomy = h - 1
            LeftTopy = (h - 1) - (crop_size * 2)
        cropped_img = img[LeftTopy:RightBottomy, LeftTopx:RightBottomx, :]
        output[idx] = cropped_img
    output = output.transpose((3, 0, 1, 2)) #l, h, w, 3 --> 3, l, h, w
    return output

def visualization(video_clip, epoch, loss, accuracy):
    vis = visdom.Visdom()
    loss_window = vis.line(X=torch.zeros((1)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='Epoch',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Train Loss']))
    vis2 = visdom.Visdom()
    acc_window = vis2.line(X=torch.zeros((1)).cpu(),
                          Y=torch.zeros((1)).cpu(),
                          opts=dict(xlabel='Epoch',
                                    ylabel='Acc',
                                    title='Validation  Accuracy',
                                    legend=['Val Acc']))
    imgs = video_clip.permute(1, 0, 2, 3)
    vis.images(imgs, win="Img")
    sleep(1)
    vis.line(X=np.column_stack(np.array([epoch])), Y=np.column_stack(np.array([loss])),
             win=loss_window, update='append')
    vis.line(X=np.array([epoch]), Y=np.array([accuracy]), win=acc_window, update='append')

def save_model(model, save_name, acc, epoch):
    torch.save(model.state_dict(), '{0}_{1}_{2:2.2f}.pt'.format(save_name, epoch, acc))