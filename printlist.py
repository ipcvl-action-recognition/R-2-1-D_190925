import os

folder_path = 'D:/Project/Pycharm/Action Recognition/Action_Recognition_v2/Test_data/Video/Val'
file_list = os.listdir(folder_path) # folder_list =  ['ApplyEyeMakeup', 'ApplyLipstick']

#print("file_list = ", file_list)

for i in file_list:
    print(i)