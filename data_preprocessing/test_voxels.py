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
    xmax, xmin,   ymax, ymin,  zmax, zmin
    8.62, 0.0018, 3.55, -5.21, 7.87, -8.19
"""

import glob
import os
import numpy as np
import csv
import time

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
    start_time = time.time()
    # print(x_res,y_res,z_res)

    for i in range(y.shape[0]):
        x_current = x_min
        x_prev = x_min
        x_count = 0
        done = False

        while x_current <= x_max and x_count < x_points and done==False:
            y_prev = y_min
            y_current = y_min
            y_count = 0
            while y_current <= y_max and y_count < y_points and done==False:
                z_prev = z_min
                z_current = z_min
                z_count = 0
                while z_current <= z_max and z_count < z_points and done==False:
                    if x[i] < x_current and y[i] < y_current and z[i] < z_current and x[i] >= x_prev and y[i] >= y_prev and z[i] >= z_prev:
                        pixel[x_count, y_count, z_count] = pixel[x_count, y_count, z_count] + 1
                        done = True

                        
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

    for i in range(0,length1):
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
        if int(frame_num[i]) in data:
            data[frame_num[i]].append([x[i], y[i], z[i]])
        else:
            data[frame_num[i]]=[]
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

    del x, y, z, data, data_pro1
    print(xmax, xmin, ymax, ymin, zmax, zmin)
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

    for i in range(0,length1):
        if wordlist[i] == "point_id:" and wordlist[i+1] == "0":
            # judge the new freame
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

    train_data = []

    i = 0
    while i+frames_together <= pixels.shape[0]:
        local_data = []
        for j in range(frames_together):
            local_data.append(pixels[i+j])

        train_data.append(local_data)
        i = i + sliding

    train_data = np.array(train_data)

    del x, y, z, data, data_pro1, pixels
    print(np.shape(train_data))
    return train_data

# parse the data file


def parse_RF_files(parent_dir, file_ext='*.txt'):
    #print(sub_dirs)
    features = np.empty((0, frames_together, 10, 32, 32))
    labels = []

    #for sub_dir in sub_dirs:
    files = sorted(glob.glob(os.path.join(parent_dir, file_ext)))
    for fn in files:
        print(fn)
        # fn is a certain file
        #print(sub_dir)
        # sub_dir is the class
        train_data = get_data(fn)
        if train_data.shape != (0,):
            features = np.vstack([features, train_data])


        for i in range(train_data.shape[0]):
            if fn[50] == '_':
                labels.append(fn[47:50])
            else:
                labels.append(fn[47:51])
        print(features.shape, len(labels))

        del train_data

    return features, labels

def parse_RF_Files_max(parent_dir, sub_dirs, file_ext='*.txt'):
    print(sub_dirs)
    features =np.empty((0, 10, 10, 32, 32) )
    labels = []
    xmax = ymax = zmax = -1
    xmin = ymin = zmin = 1000
    for sub_dir in sub_dirs:
        files=sorted(glob.glob(os.path.join(parent_dir,sub_dir, file_ext)))
        for fn in files:
            print(fn)
            print(sub_dir)
            xmax_, xmin_, ymax_, ymin_, zmax_, zmin_ = get_biggest(fn)
            xmax = max(xmax, xmax_)
            xmin = min(xmin, xmin_)
            ymax = max(ymax, ymax_)
            ymin = min(ymin, ymin_)
            zmax = max(zmax, zmax_)
            zmin = min(zmin, zmin_)
            

    return xmax, xmin, ymax, ymin, zmax, zmin

if __name__=="__main__":
    parent_dir = 'D:/研究生/研一/跌倒检测/1017/data_6.16_dynamic_txt/test'
    sub_dirs = ['fall', 'lie', 'sit', 'walk']
    extract_path = 'D:/研究生/研一/跌倒检测/1017/data_6.16_dynamic_txt/test_voxel_'
    # xmax = ymax = zmax = -1
    # xmin = ymin = zmin = 1000
    # f = np.array([])
    # l = np.array([])
    #for sub_dir in sub_dirs:
    features, labels = parse_RF_files(parent_dir)
    #print(features.shape)
    # f = np.append(f, features)
    # l = np.append(l, labels)
    Data_path = extract_path + "data"
    np.savez(Data_path, features, labels)
    del features, labels

