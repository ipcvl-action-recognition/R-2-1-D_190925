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
        self.shape = (self.length, self.height, self.width)
    def get_shape(self):
        return self.length, self.height, self.width
def transformation_3D_to_2D(buffer):
    first = True
    if len(buffer.shape) == 5:
        b, c, l, h, w = buffer.shape

        # print(buffer[0, :, 0, 0, 0], buffer[0, :, 1, 0, 0])
        # print(buffer.shape)
        buffer = buffer.permute((0, 2, 1, 3, 4))
        # buffer = np.transpose(buffer, (0, 2, 1, 3, 4))
        buffer = buffer.reshape((b, -1, h, w))
        # print(buffer[0, 0:3, 0, 0], buffer[0, 3:6, 0, 0])
    else:
        c, l, h, w = buffer.shape
        buffer = buffer.permute((1, 0, 2, 3))
        buffer = buffer.reshape((-1, h, w))
    return buffer


def load_video(path, video_name, resize_resolution):
    cap = cv2.VideoCapture(path+video_name)
    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buffer = torch.empty((frame_length, resize_resolution.height, resize_resolution.width, 3), dtype=torch.float32).cuda()
    origin_resolution = resolution(frame_length, frame_height, frame_width)
    count = 0
    while count < frame_length and cap.isOpened():
        ret, frame = cap.read()  # frame 읽어오면 ret = true 실패하면 ret = false
        if ret:
            if (frame_height != resize_resolution.height) or (frame_width != resize_resolution.width):
                frame = cv2.resize(frame, (resize_resolution.width, resize_resolution.height))  # frame_resize w, h
            buffer[count] = torch.from_numpy(frame).cuda()
            count += 1
        else:
            break
    cap.release()
    # buffer = buffer.transpose((3, 0, 1, 2)) # 3, l, h, w
    return buffer, origin_resolution

def video_resize(buffer, resolution):
    for idx, frame in enumerate(buffer):
        frame = cv2.resize(frame, (resolution.width, resolution.height))
        buffer[idx] = frame
    return buffer



def pre_processing(buffer, resolution, crop_size):
    buffer = buffer/255
    # buffer = crop(buffer, clip_len=16, crop_size=crop_size)
    return buffer

def crop(buffer, label, clip_len):
    time_index = np.random.randint(buffer.shape[1] - clip_len) # 196-8
    # print("time_index = ", time_index ) # 186
    label = label[time_index:time_index+clip_len]
    buffer = buffer[:, time_index:time_index + clip_len, :, :].cuda()
    # print(buffer.shape, label)
    return buffer, label

def model_test(clip, model):
    # input = np.transpose(clip, (3, 0, 1, 2))
    # input = torch.from_numpy(clip).cuda()
    with torch.no_grad():
        output = model(clip).cuda()
    output = torch.sigmoid(output).cpu().squeeze()
    y_pred_index = torch.round(output).int()
    return y_pred_index

def count_y_pred(y_pred_index, fall_down, none):
    if y_pred_index == 0:
        none += 1
    else:
        fall_down += 1
    return fall_down, none

#####  Label Function  #####


# def coordinates_resize(input, resolution):


def load_label_file(label_file, origin_resolution, resize_resolution):
    label_list = []
    person_label_box = []
    readlabel = open(label_file, 'r')

    for i, line in enumerate(readlabel):
        line = line.strip()
        line = line.split()
        label_list.append(line[1])
        person_label_box.append(coordinates_change(line, origin_resolution, resize_resolution))
        # person_label_box.append((line[2], line[3], line[4], line[5]))
    return label_list, person_label_box

def coordinates_change(abs, origin_resolution, resize_resolution):
    w_ratio = resize_resolution.width / origin_resolution.width
    h_ratio = resize_resolution.height / origin_resolution.height
    return (float(abs[2]) * w_ratio / resize_resolution.width, float(abs[3]) * h_ratio / resize_resolution.height,
            float(abs[4]) * w_ratio / resize_resolution.width, float(abs[5]) * h_ratio / resize_resolution.height)

def correct_falldown_count(label_list):
    fall_down = 0
    none = 0
    for one in label_list:
        if one == 1:
            fall_down += 1
        else:
            none += 1
    return fall_down, none

def crop_video_from_label(video, origin_resolution, person_label_box, crop_size):  # Input Video --> ROI Crop 224x224
    # video = video.transpose((1, 2, 3, 0))  # 3, l, h, w --> l ,h, w, 3
    l, h, w, c = video.shape
    # _, h, w = origin_resolution.get_shape()
    output = torch.empty((l, crop_size*2, crop_size*2, 3), dtype=torch.float32).cuda()
    for idx, img in enumerate(video):
        if person_label_box[idx] is None:
            person_x = 0
            person_y = 0
            person_w = 0
            person_h = 0
            center_x = 0
            center_y = 0
        else:
            person_x = round((float(person_label_box[idx][0]) * w))
            person_y = round((float(person_label_box[idx][1]) * h))
            person_w = round((float(person_label_box[idx][2]) * w))
            person_h = round((float(person_label_box[idx][3]) * h))
            center_x = int((person_x + person_w / 2))
            center_y = int((person_y + person_h / 2))
            # print(person_label_box[idx][0], w)
            # print(person_x, person_y, person_w, person_h, center_x, center_y)
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
        # print(LeftTopx, RightBottomx, LeftTopy, RightBottomy)
        cropped_img = img[LeftTopy:RightBottomy, LeftTopx:RightBottomx, :]
        # print(img[100, 100, :])
        # temp2 = cropped_img.data.cpu().numpy().astype(np.uint8)
        # cv2.imshow("asdf111", temp2)
        # cv2.waitKey(0)
        output[idx] = cropped_img
    output = output.permute((3, 0, 1, 2))
    # output = output.transpose((3, 0, 1, 2)) #l, h, w, 3 --> 3, l, h, w
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