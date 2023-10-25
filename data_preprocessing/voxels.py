"""
Extract the VOXEL representation from the point cloud dataset

USAGE: change the parent_dir and extract_path to the correct variables.

- parent_dir is raw Path_to_training_or_test_data.
- extract_path is the Path_to_put_extracted_data samples.

EXAMPLE: SPECIFICATION

parent_dir = '/Users/sandeep/Research/Ti-mmWave/data/Temp_Samples/Train/'
sub_dirs=['boxing','jack','jump','squats','walk']
extract_path = '/Users/sandeep/Research/Ti-mmWave/data/extract/Train_Data_voxels_'
"""

"""
    x_max = 6

    x_min = -5.13

    y_max = 7.14

    y_min = 0.28

    z_max = 2.31

    z_min = -2.37
"""

import glob
import os
import numpy as np
import csv
import time


# xmax = ymax = zmax = -1
# xmin = ymin = zmin = 1000
# minmin = 10000
frames_together = 20
sliding = 10

def voxalize(x_points, y_points, z_points, x, y, z):
    """
        for a frame
    """
    x_min = np.min(x)
    x_max = np.max(x)

    y_min = np.min(y)
    y_max = np.max(y)

    z_min = np.min(z)
    z_max = np.max(z)

    z_res = (z_max - z_min)/z_points
    y_res = (y_max - y_min)/y_points
    x_res = (x_max - x_min)/x_points

    pixel = np.zeros([x_points, y_points, z_points])

    x_current = x_min
    y_current = y_min
    z_current = z_min

    x_prev = x_min
    y_prev = y_min
    z_prev = z_min

    x_count = 0
    y_count = 0
    z_count = 0
    # start_time = time.time()

    for i in range(y.shape[0]):
        x_current = x_min
        x_prev = x_min
        x_count = 0
        done = False

        while x_current <= x_max and x_count < x_points and done == False:
            y_prev = y_min
            y_current = y_min
            y_count = 0
            while y_current <= y_max and y_count < y_points and done == False:
                z_prev = z_min
                z_current = z_min
                z_count = 0
                while z_current <= z_max and z_count < z_points and done==False:
                    if x[i] < x_current and y[i] < y_current and z[i] < z_current and x[i] >= x_prev \
                            and y[i] >= y_prev and z[i] >= z_prev:
                        pixel[x_count, y_count, z_count] = pixel[x_count, y_count, z_count] + 1
                        done = True

                        #velocity_voxel[x_count,y_count,z_count] = velocity_voxel[x_count,y_count,z_count] + velocity[i]
                    z_prev = z_current
                    z_current = z_current + z_res
                    z_count = z_count + 1
                y_prev = y_current
                y_current = y_current + y_res
                y_count = y_count + 1
            x_prev = x_current
            x_current = x_current + x_res
            x_count = x_count + 1
    # for i in range(len(pixel)):
    #     for j in range(len(pixel[i])):
    #         if pixel[i][j].any()!=0:
    #             print("!!!")
            # print(pixel[i][j])
    # print(np.shape(pixel))
    return pixel

def get_biggest(file_path):
    """
        get the xmax, xmin, ymax, ymin, zmax, zmin of the file: file_path
    """
    with open(file_path) as f:
        lines = f.readlines()

    frame_num_count = -1
    frame_num = []
    x = []
    y = []
    z = []
    wordlist = []


    for x1 in lines:
        for word in x1.split():
            wordlist.append(word)

    length1 = len(wordlist)

    for i in range(0, length1):
        if wordlist[i] == "point_id:" and wordlist[i+1] == "0":
            frame_num_count += 1
        if wordlist[i] == "point_id:":
            frame_num.append(frame_num_count)
        if wordlist[i] == "x:":
            x.append(wordlist[i+1])
        if wordlist[i] == "y:":
            y.append(wordlist[i+1])
        if wordlist[i] == "z:":
            z.append(wordlist[i+1])

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    frame_num = np.asarray(frame_num)

    x = x.astype(float)
    y = y.astype(float)
    z = z.astype(float)
    frame_num = frame_num.astype(int)


    data = dict()
    for i in range(len(frame_num)):
        # print(data)
        if int(frame_num[i]) in data:
            data[frame_num[i]].append([x[i], y[i], z[i]])
        else:
            data[frame_num[i]] = []
            # print(data)
            data[frame_num[i]].append([x[i], y[i], z[i]])
    # print("data:")
    # print(data)
    data_pro1 = dict()

    # Merging of frames together with sliding of  frames
    together_frames = 1
    sliding_frames = 1

    #we have frames in data
    frames_number = []
    for i in data:
        frames_number.append(i)

    frames_number = np.array(frames_number)
    total_frames = frames_number.max()

    i = 0
    j = 0

    while i < total_frames-1:
        data_pro1[j] = data[i]
        j = j+1
        i = i+1

    xmax = ymax = zmax = -1
    xmin = ymin = zmin = 1000

    # Now for 2 second windows, we need to club together the frames and we will have some sliding windows
    for i in data_pro1:
        f = data_pro1[i]
        f = np.array(f)

        #y and z points in this cluster of frames
        x_c = f[:, 0]
        y_c = f[:, 1]
        z_c = f[:, 2]
        xmax = max(xmax, np.max(x_c))
        xmin = min(xmin, np.min(x_c))
        ymax = max(ymax, np.max(y_c))
        ymin = min(ymin, np.min(y_c))
        zmax = max(zmax, np.max(z_c))
        zmin = min(zmin, np.min(z_c))

    del x, y, z,  data, data_pro1
    #print(xmax, xmin, ymax, ymin, zmax, zmin)
    return xmax, xmin, ymax, ymin, zmax, zmin



def get_data(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    frame_num_count = -1
    frame_num = []
    x = []
    y = []
    z = []
    wordlist = []


    for x1 in lines:
        for word in x1.split():
            wordlist.append(word)

    length1 = len(wordlist)
    #print(wordlist)

    for i in range(0, length1):
        if wordlist[i] == "point_id:" and wordlist[i+1] == "0":
            # judge the new frame
            frame_num_count += 1
        if wordlist[i] == "point_id:":
            frame_num.append(frame_num_count)
        if wordlist[i] == "x:":
            x.append(wordlist[i+1])
        if wordlist[i] == "y:":
            y.append(wordlist[i+1])
        if wordlist[i] == "z:":
            z.append(wordlist[i+1])

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    frame_num = np.asarray(frame_num)

    x = x.astype(float)
    y = y.astype(float)
    z = z.astype(float)
    frame_num = frame_num.astype(int)

    #print(frame_num)


    data = dict()

    for i in range(len(frame_num)):
        if int(frame_num[i]) in data:
            data[frame_num[i]].append([x[i], y[i], z[i]])
        else:
            data[frame_num[i]] = []
            data[frame_num[i]].append([x[i], y[i], z[i]])

    data_pro1 = dict()

    # Merging of frames together with sliding of  frames
    together_frames = 1
    sliding_frames = 1

    #we have frames in data
    frames_number = []
    for i in data:
        frames_number.append(i)

    frames_number = np.array(frames_number)
    total_frames = frames_number.max()

    i = 0
    j = 0

    while together_frames+i < total_frames:

        curr_j_data = []
        for k in range(together_frames):
            curr_j_data = curr_j_data + data[i+k]
        #print(len(curr_j_data))
        data_pro1[j] = curr_j_data
        j = j + 1
        i = i + sliding_frames

    pixels = []

    # Now for 2 second windows, we need to club together the frames and we will have some sliding windows
    for i in data_pro1:
        f = data_pro1[i]
        f = np.array(f)

        #y and z points in this cluster of frames
        x_c = f[:, 0]
        y_c = f[:, 1]
        z_c = f[:, 2]

        pix = voxalize(10, 32, 32, x_c, y_c, z_c)
        #print(i, f.shape,pix.shape)
        pixels.append(pix)

    pixels = np.array(pixels)
    #print(pixels.shape[0])

    train_data = []

    i = 0
    while i + frames_together <= pixels.shape[0]:
        local_data = []
        for j in range(frames_together):
            local_data.append(pixels[i+j])

        train_data.append(local_data)
        i = i + sliding

    train_data = np.array(train_data)

    del x, y, z, data, data_pro1, pixels

    return train_data



def get_frames(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    frame_num_count = -1

    wordlist = []


    for x1 in lines:
        for word in x1.split():
            wordlist.append(word)

    length1 = len(wordlist)

    for i in range(0, length1):
        if wordlist[i] == "point_id:" and wordlist[i+1] == "0":
            # judge the new freame
            frame_num_count += 1
    return frame_num_count

# parse the data file
def parse_RF_files(parent_dir, sub_dirs, file_ext='*.txt'):
    #print(sub_dirs)
    features = np.empty((0, frames_together, 10, 32, 32))
    labels = []

    for sub_dir in sub_dirs:
        files = sorted(glob.glob(os.path.join(parent_dir, sub_dir, file_ext)))
        #print(len(files))
        for fn in files:
            #print(fn)
            # fn is a certain file
            #print(sub_dir)
            # sub_dir is the class
            train_data = get_data(fn)
            #print(features.shape, train_data.shape)
            if train_data.shape != (0,):
                features = np.vstack([features, train_data])


            for i in range(train_data.shape[0]):
                #print(train_data.shape)
                labels.append(sub_dir)
            #print(features.shape, len(labels))

            del train_data

    return features, labels

def parse_RF_Files_max(parent_dir, sub_dirs, file_ext='*.txt'):
    # get the extreme coordinate
    #print(sub_dirs)
    features = np.empty((0, 60, 10, 32, 32))
    labels = []
    xmax = ymax = zmax = -1
    xmin = ymin = zmin = 1000
    for sub_dir in sub_dirs:
        files = sorted(glob.glob(os.path.join(parent_dir, sub_dir, file_ext)))
        for fn in files:
            #print(fn)
            #print(sub_dir)
            xmax_, xmin_, ymax_, ymin_, zmax_, zmin_ = get_biggest(fn)
            xmax = max(xmax, xmax_)
            xmin = min(xmin, xmin_)
            ymax = max(ymax, ymax_)
            ymin = min(ymin, ymin_)
            zmax = max(zmax, zmax_)
            zmin = min(zmin, zmin_)


    return xmax, xmin, ymax, ymin, zmax, zmin

def least_frames(parent_dir, sub_dirs, file_ext='*.txt'):
    #print(sub_dirs)

    minmin = 10000
    for sub_dir in sub_dirs:
        files = sorted(glob.glob(os.path.join(parent_dir,sub_dir, file_ext)))
        for fn in files:
            #print(fn)
            # fn is a certain file
            #print(sub_dir)
            # sub_dir is the class
            frames = get_frames(fn)
            #print(frames)
            minmin = min(minmin, frames)

    return minmin

if __name__=="__main__":
    parent_dir = 'D:/研究生/研一/跌倒检测/1017/data_6.16_dynamic_txt/train'
    sub_dirs = ['fall', 'lie', 'sit', 'walk']
    extract_path = 'D:/研究生/研一/跌倒检测/1017/data_6.16_dynamic_txt/train_voxel_'


    # get the extreme coordinates
    # for sub_dir in sub_dirs:
    #     xmax_, xmin_, ymax_, ymin_, zmax_, zmin_ = parse_RF_Files_max(parent_dir, [sub_dir])
    #     xmax = max(xmax, xmax_)
    #     xmin = min(xmin, xmin_)
    #     ymax = max(ymax, ymax_)
    #     ymin = min(ymin, ymin_)
    #     zmax = max(zmax, zmax_)
    #     zmin = min(zmin, zmin_)

    # # get the least frame
    #     m=least_frames(parent_dir,[sub_dir])
    #     minmin=min(minmin, m)
    #     print(m)
    # print(minmin)



    # for train
    for sub_dir in sub_dirs:
        features, labels = parse_RF_files(parent_dir, [sub_dir])
        Data_path = extract_path + sub_dir
        np.savez(Data_path, features, labels)
        del features, labels



    # for test
    # f=np.array([])
    # l=np.array([])
    # for sub_dir in sub_dirs:
    #     features, labels = parse_RF_files(parent_dir,[sub_dir])
    #     f=np.append(f,features)
    #     l=np.append(l,labels)
    #     del features,labels
    # Data_path = extract_path + "data"
    # np.savez(Data_path, f,l)
    # del f,l