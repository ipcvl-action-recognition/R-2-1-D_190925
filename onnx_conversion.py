import torch
import torchvision
import numpy as np
from network import C3DNet
import cv2
from torch import nn
import R2Plus1D_model
torch.backends.cudnn.benchmark = True
'''

frames = []
cap = cv2.VideoCapture('./Test_data/Video/Train/Coffee_room_01 (02).mp4')
retaining =True
count=0
model = R2Plus1D_model.R2Plus1DClassifier(num_classes=1, layer_sizes=(2, 2, 2, 2)).cuda()
checkpoint = torch.load('Weights/Falldown_0816_best_acc_21_0.18_95.58.pt', map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint)
model.eval()

while retaining:
    retaining, frame = cap.read()
    cv2.imshow("asdf", frame)
    cv2.waitKey(1)
    tmp = cv2.resize(frame, (224, 224))
    tmp = tmp / 255.0
    frames.append(tmp)
    if len(frames)==16:
        np_frames = np.array(frames)
        np_frames = np.transpose(np_frames, (3, 0, 1, 2))

        tc_frames = torch.from_numpy(np_frames)
        print(tc_frames.shape)
        output = model(tc_frames)
        print(output)
        # tc_frames = tc_frames.view(-1)
        # print(tc_frames.shape)
        # for k in range(100*3,2408448,150528):
        #     print(tc_frames[k:k+3])
        # for k in range(224*100*3+100*3,2408448,150528):
        #     print(tc_frames[k:k+3])
        frames.pop(0)
        count+=1
        if count==2:
            retaining = False
            break
'''
model = R2Plus1D_model.R2Plus1DClassifier(num_classes=1, layer_sizes=(2, 2, 2, 2)).cuda()
checkpoint = torch.load('avgpool177.pt', map_location=lambda storage, loc: storage)
# model.load_state_dict(checkpoint)
model.eval()
dummy_input = torch.randn(1, 3, 16, 224, 224).cuda()
# input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]
torch.onnx.export(model, dummy_input, "adaptive1.onnx", verbose=True, output_names=output_names)
# torch.onnx.export(model, dummy_input, "R2plus1D_model.onnx", verbose=False)
