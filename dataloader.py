import numpy as np
import cv2
import torch
import xml.etree.ElementTree as elemTree
import os
from torch.utils.data import DataLoader
from ImageArg import ImageArg
import time
import utils
from Setting import TrainSetting
class UCF101_Dataset:
    def __init__(self, folder_path, readfile, is_train=True):
        self.folder_path = folder_path
        folder_list = os.listdir(folder_path)
        self.folder_list = folder_list

        data_list = []
        file = open(readfile, 'r')

        for line in file:
            line = line.split(' ')[0].strip()
            line = folder_path.strip() + '/' + line
            data_list.append(line)

        self.video_list = data_list
        self.clip_len = 16
        self.crop_size = 112
        self.resize_width = 171
        self.resize_height = 128
        self.data_length = len(self.video_list)

        self.imgArg = ImageArg()
        self.allImage_buffer_list = []
        self.is_train = is_train
        # self.loadAllVideo()

    def loadAllVideo(self):
        for idx in range(len(self.video_list)):
            self.allImage_buffer_list.append([self.loadvideo(self.video_list[idx])])

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        buffer = self.loadvideo(self.video_list[idx])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        buffer = self.normalize(buffer)
        label = self.video_list[idx].split('/')[-2]
        label = self.folder_list.index(label)
        return torch.from_numpy(buffer), label

    def loadvideo(self, fname):
        cap = cv2.VideoCapture(fname)
        if not cap.isOpened():
            print("File is Not Open!!")
        frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # buffer = np.empty((frame_length, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        buffer = np.array((frame_length, self.resize_height, self.resize_width, 3), np.dtype('int8'))

        count = 0
        # print(frame_length)
        while count < frame_length and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                buffer[count] = frame
                count += 1

            else:
                break

        cap.release()
        buffer = buffer.transpose((3, 0, 1, 2))  # c, l, h, w
        return buffer

    def crop(self, buffer, clip_len, crop_size):
        time_index = np.random.randint(buffer.shape[1] - clip_len)
        height_index = np.random.randint(buffer.shape[2] - crop_size)
        width_index = np.random.randint(buffer.shape[3] - crop_size)
        # print(width_index, height_index)
        buffer = buffer[:, time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size]

        return buffer

    def normalize(self, buffer):
        # buffer = buffer.transpose((1, 2, 3, 0))  # f, h, w, c
        buffer = buffer.astype(np.float32)
        # for i, frame in enumerate(buffer):
        #    frame -= np.array([[[90.0, 98.0, 102.0]]])
        #    buffer[i] = frame
        buffer = buffer / 255.0  # (buffer-128)/128
        # buffer = buffer.transpose((3, 0, 1, 2))
        return buffer

    def label_extraction(self, fname, startFrame):
        label_list = self.framefile_path + "/" + fname
        readlabel = open(label_list, 'r')
        label_list = []
        for i, line in enumerate(readlabel):
            line = line.strip()
            line = line.split(' ')
            label_list.append(line[1])

        if self.is_train:
            label_list = label_list[startFrame:startFrame + self.clip_len]
            label = int(label_list[-1])
            return label
        else:
            return [''.join(label_list)]

class Le2i_VideoDataset:
    def __init__(self, TrainSetting, resolution, is_train):
        self.resolution = resolution
        self.folder_path = TrainSetting.folder_path
        folder_list = os.listdir(TrainSetting.folder_path)
        self.folder_list = folder_list

        data_list = []
        file_list = open(TrainSetting.readfile, 'r')
        for line in file_list:
            line = line.strip()
            data_list.append(TrainSetting.folder_path + "/" + line)

        self.framefile_path = TrainSetting.framefile # framefile = ./Test_data/label_Result2/Train
        frame_folder_list = os.listdir(TrainSetting.framefile)
        self.frame_label_list = frame_folder_list # frame_folder_list = ['Coffee_room_01 (02).txt']
        self.video_list = data_list # data_list = ['./Test_data/Video2/Train/Coffee_room_01 (02).mp4']
        self.clip_len = TrainSetting.clip_len
        self.crop_size = TrainSetting.crop_size
        self.crop_size_width = TrainSetting.crop_size
        self.crop_size_height = TrainSetting.crop_size
        self.resize_width = resolution.width
        self.resize_height = resolution.height
        self.data_length = len(self.video_list)
        self.imgArg = ImageArg()
        self.allImage_buffer_list = []
        self.is_train = is_train

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        if (self.is_train==True):
            print(self.folder_path+self.frame_label_list[idx])
            label_list, person_label_box = utils.load_label_file(self.framefile_path+self.frame_label_list[idx], self.resolution)
            filename = self.video_list[idx].split('/')[-1]
            buffer = utils.load_video(self.folder_path, filename, self.resolution)
            buffer = utils.crop_video_from_label(buffer, person_label_box, self.crop_size)
            buffer = self.normalize(buffer)
            if label_list.count('1') >= int(len(label_list) / 2):
                label = np.array(1)
                # print(label)
            else:
                label = np.array(0)
            return torch.from_numpy(buffer), torch.from_numpy(label)
        if (self.is_train == False):
            return self.video_list[idx], (self.framefile_path + "/" +self.frame_label_list[idx])


    def loadvideo(self, fname, label_list, label_person_box):
        cap = cv2.VideoCapture(fname)
        if not cap.isOpened():
            print("File is Not Open!!")
        frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        no_person_idx = []
        for i in range(frame_length):
            if label_person_box[i] == ('0', '0', '0', '0'):
                no_person_idx.append(i)

        for i in reversed(no_person_idx):
            del label_list[i]
            del label_person_box[i]

        buffer = np.empty((frame_length-len(no_person_idx), self.resize_height, self.resize_width, 3), np.dtype('float32'))

        count = 0
        buffer_cnt = 0
        while count < frame_length and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if count not in no_person_idx:
                    # print("before = ", frame.shape)
                    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                        frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                    # print("after = ", frame.shape)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    buffer[buffer_cnt] = frame
                    buffer_cnt += 1
                count += 1

            else:
                break
        cap.release()
        buffer = buffer.transpose((3, 0, 1, 2))  # c, l, h, w
        return buffer, label_list, label_person_box, frame_width, frame_height

    def crop(self, buffer, clip_len, label_list, label_person_box, frame_width, frame_height):
        time_index = np.random.randint(buffer.shape[1] - clip_len)
        crop_size_height = self.crop_size_height
        crop_size_width = self.crop_size_width

        resize_width = self.resize_width
        resize_height = self.resize_height

        temp = np.empty((3, clip_len, crop_size_height * 2, crop_size_width * 2), np.dtype('float32'))

        buffer = buffer[:, time_index:time_index + clip_len]

        for i in range(clip_len):
            # print("label_person_box[time_index + i] = ", label_person_box[time_index + i])
            # center_x = int(int(label_person_box[time_index + i][0])) + int(
            #     (int(label_person_box[time_index + i][2]) / 2))  # x + w/2 중점
            # center_y = int(int(label_person_box[time_index + i][1])) + int(
            #     (int(label_person_box[time_index + i][3]) / 2))  # y + h/2

            person_x = round((float(label_person_box[time_index + i][0]) / float(frame_width)) * resize_width)
            person_y = round((float(label_person_box[time_index + i][1]) / float(frame_height)) * resize_height)
            person_w = round((float(label_person_box[time_index + i][2]) / float(frame_width)) * resize_width)
            person_h = round((float(label_person_box[time_index + i][3]) / float(frame_height)) * resize_height)

            center_x = int(person_x + int(person_w / 2))
            center_y = int(person_y + int(person_h / 2))

            LeftTopx = center_x - crop_size_width
            LeftTopy = center_y - crop_size_height

            RightBottomx = center_x + crop_size_width
            RightBottomy = center_y + crop_size_height


            if(LeftTopx < 0) :
                LeftTopx = 0
                RightBottomx = (crop_size_width * 2)
            if (LeftTopy < 0):
                LeftTopy = 0
                RightBottomy = (crop_size_height * 2)

            if (RightBottomx > buffer.shape[3] - 1):
                RightBottomx = buffer.shape[3] - 1
                LeftTopx = (buffer.shape[3] - 1) - (crop_size_width * 2)

            if (RightBottomy > buffer.shape[2] - 1):
                RightBottomy = buffer.shape[2] - 1
                LeftTopy = (buffer.shape[2] - 1) - (crop_size_height * 2)

            # print("LT : {}, {}, RT : {}, {}, Size : {}, {}".format(LeftTopx, LeftTopy, RightBottomx, RightBottomy, RightBottomx-LeftTopx,  RightBottomy-LeftTopy))

            temp[:, i] = buffer[:, i, LeftTopy : RightBottomy , LeftTopx : RightBottomx]

            # ### image 확인
            # orgImage = buffer[:, i]
            # orgImage = orgImage.astype("uint8")
            # orgImage = orgImage.transpose((1, 2, 0))  # c, l, h, w
            # orgImage = cv2.cvtColor(orgImage, cv2.COLOR_RGB2BGR)
            # orgImage = cv2.rectangle(orgImage, (LeftTopx, LeftTopy), (RightBottomx, RightBottomy), (255, 0, 0), 3)
            # # print(temp[:, i].shape)
            #
            # curFrame = temp[:, i]
            # f = curFrame.astype("uint8")
            # # print(f.shape)
            # tt = f.transpose((1, 2, 0))  # c, l, h, w
            # tt = cv2.cvtColor(tt, cv2.COLOR_RGB2BGR)
            # # print(tt.shape)
            # cv2.imshow("Test", tt)
            # cv2.imshow("Test2", orgImage)
            # cv2.waitKey()

        buffer = temp
        # print("time_index = ", time_index)
        label_list = label_list[time_index:time_index + clip_len]
        # print("label_list = ", label_list)
        label_person_box = label_person_box[time_index:time_index + clip_len]
        # print("label_person_box = ", label_person_box)
        return buffer, label_list, label_person_box


    def normalize(self, buffer):
        buffer = buffer.astype(np.float32)
        buffer = buffer / 255.0  # (buffer-128)/128
        # buffer = buffer.transpose((1, 2, 3, 0))  # f, h, w, c
        # for i, frame in enumerate(buffer):
        #     frame -= np.array([[[90, 98, 102]]]).astype('float32')
        #     buffer[i] = frame
        # buffer = buffer.transpose((3, 0, 1, 2))
        return buffer

    def label_extraction(self, fname):
        label_list = self.framefile_path + "/" + fname
        readlabel = open(label_list, 'r')

        label_list = []
        label_person_box = []
        for i, line in enumerate(readlabel):
            line = line.strip()
            line = line.split(' ')
            label_list.append(line[1])
            label_person_box.append((line[2], line[3], line[4], line[5]))

        # print(label_person_box[0])
        return label_list, label_person_box