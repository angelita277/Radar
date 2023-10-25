import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os


def func_ReadData(Data):
    # print(type(Data[0]))
    zm = []
    for i in range(len(Data)):
        # print(type(Data[i]["snr"]))
        zm.append([Data[i]["snr"]*0.04, Data[i]["range"]*0.00025, Data[i]["azimuth"]*0.01*(180/math.pi), Data[i]["doppler"]*0.00028, Data[i]["elevation"]*0.01*(180/math.pi)])
        # zm.append([Data[i]["snr"]*0.04, Data[i]["range"]*0.00025, 0.005*Data[i]["range"]*np.sin(np.deg2rad(
        #     Data[i]["elevation"])), Data[i]["doppler"]*0.00028, Data[i]["elevation"]*0.01*(180/math.pi)])

    return np.array(zm)

def read_xyz(Data):
    zm=[]
    for i in range(len(Data)):
        _range = Data[i]["range"]*0.00025
        _azimuth = Data[i]["azimuth"] * 0.01
        _elevation = Data[i]["elevation"] * 0.01

        x = _range * np.cos(_elevation) * np.sin(_azimuth)
        y = _range * np.cos(_elevation) * np.cos(_azimuth)
        z = _range * np.sin(_elevation)
        zm.append([x, y, z])
    return np.array(zm)

def csv_2_txt(csv_file, txt_file):
    # each file is a motion sequence
    df = pd.read_csv(csv_file, header=0)

    #points contains the data of the sequence
    points = df["points"]
    points = np.array(points)
    
    with open(txt_file, "w") as f:

        #counts each frame of the sequence
        for i in range(len(points)):
            data = eval(points[i])
            coor = read_xyz(data)
            # print(coor)
            
            for i in range(len(coor)):
                if i == 0:
                    f.write("point_id: 0\n")
                else:
                    f.write("point_id: 1\n")
                f.write("x: " + str(coor[i][0]) + "\n")
                f.write("y: " + str(coor[i][1]) + "\n")
                f.write("z: " + str(coor[i][2]) + "\n")
        f.close()
    return




if __name__ == "__main__":
    path = "D:/研究生/研一/跌倒检测/1017/data_6.16_dynamic"
    root_txt = ""
    p = os.walk(path)
    for root, dirs, files in p:
        for file in files:
            if file == ".DS_Store" or file[-3:] != "csv":
                continue

            if (len(root) == 42):
                root_txt = root[:-5] + "_txt" + root[-5:]
                #print(root_txt)
            else:
                root_txt = root[:-4] + "_txt" + root[-4:]
                #print(root_txt)

            a=str(os.path.join(root, file))
            a=a[:-3]
            origin_path = a + "csv"
            after_path = a[:37] + "_txt" + a[37:] + "txt"

            if not os.path.exists(root_txt):
                os.makedirs(root_txt)
            #print(origin_path, after_path)
            csv_2_txt(csv_file=origin_path, txt_file=after_path)