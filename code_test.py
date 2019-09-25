import cv2
import numpy as np
import os


video_list = 'D:/Project/Data Set/action_recognition/UCF101/UCF-101/YoYo/v_YoYo_g25_c05.avi'
folder_path = 'D:/Project/Data Set/action_recognition/UCF101/UCF-101'


### Label 과정
# folder_list = os.listdir(folder_path)
# print("folder_list = " , folder_list)
# print("folder_list_len = " , len(folder_list))
#
# print("video_list.split('/') = " , video_list.split('/'))
# label = video_list.split('/')[-2]
# print("label = " , label)
# one_hot_vector = np.eye(len(folder_list))
#
# label = one_hot_vector[folder_list.index(label)]
# print("label_one_hot = " , label )
# index = np.where(label == 1)
# print("indepx = " , index)


### Buffer 과정
cap = cv2.VideoCapture(video_list)
if not cap.isOpened():
    print("File is Not Open!!")
frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
buffer = np.empty((frame_length, 128, 171, 3),np.dtype('float32'))  # buffer = [length, h, w, 3]
print("buffer_shape = ", buffer.shape)

count = 0

while count < frame_length and cap.isOpened():
    ret, frame = cap.read()  # frame 읽어오면 ret = true 실패하면 ret = false
    # print("frame_shape = ", frame.shape) # (240, 320, 3) h w c
    # print("frame_w = ", frame_width) # 320
    # print("frame_h = ", frame_height) # 240
    if ret:
        if (frame_height != 128) or (frame_width != 171):
            frame = cv2.resize(frame, (171, 128))  # frame_resize w, h
        # print("frame_shape = ", frame.shape) # (128, 171, 3) h w c
        cv2.imshow("frame", frame)
        cv2.waitKey(33)
        buffer[count] = frame
        count += 1
    else:
        break

cap.release()
print("buffer_shape = ", buffer.shape) # (l, h, w, c)
buffer = buffer.transpose((3, 0, 1, 2))  # c, l, w, h
print("buffer_shape = ", buffer.shape) # (c, l, h, w)


clip_len = 8
crop_size = 112
def crop(buffer, clip_len, crop_size):
    time_index = np.random.randint(buffer.shape[1] - clip_len) # 196-8
    print("time_index = ", time_index ) # 186
    height_index = np.random.randint(buffer.shape[2] - crop_size)
    print("height_index = ", height_index) # 3
    width_index = np.random.randint(buffer.shape[3] - crop_size)
    print("width_index = ", width_index) # 5

    buffer = buffer[:, time_index:time_index + clip_len,
             height_index:height_index + crop_size,
             width_index:width_index + crop_size]
    return buffer

def normalize(buffer):
    buffer = (buffer-128)/128
    return buffer

buffer = crop(buffer, clip_len, crop_size)
print("buffer_shape = ", buffer.shape) # (c, l, h, w)
buffer = normalize(buffer)
print("buffer_shape = ", buffer.shape) # (c, l, h, w)