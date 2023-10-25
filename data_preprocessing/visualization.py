import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os


def func_ReadData(Data):
    # print(type(Data[0]))
    zm = []
    for i in range(len(Data)):
        # print(type(Data[i]["snr"]))
        # zm.append([Data[i]["snr"]*0.04, Data[i]["range"]*0.00025, Data[i]["azimuth"]*0.01*(180/math.pi), Data[i]["doppler"]*0.00028, Data[i]["elevation"]*0.01*(180/math.pi)])
        zm.append([Data[i]["snr"]*0.04, Data[i]["range"]*0.00025, 0.005*Data[i]["range"]*np.sin(np.deg2rad(
            Data[i]["elevation"])), Data[i]["doppler"]*0.00028, Data[i]["elevation"]*0.01*(180/math.pi)])

    return np.array(zm)


if __name__ == "__main__":
    frame_together = 20
    sliding = 10

    src_dir_path = "D:/研究生/研一/跌倒检测/1017/data_6.16_dynamic/"
    sub_dir = ['fall', 'lie', 'sit', 'walk']
    dst_dir_path = "D:/研究生/研一/跌倒检测/1017/visualize/"

    for sd in sub_dir:
        full_src_path = src_dir_path + sd
        files = [os.path.join(full_src_path, f) for f in os.listdir(full_src_path)]

        #print(files[1][-24:-4])

        for file in files:
            if file == ".DS_Store" or file[-3:] != "csv":
                continue

            if not os.path.exists(dst_dir_path + sd):
                os.makedirs(dst_dir_path + sd)

            if len(sd) == 4:
                save_path = dst_dir_path + sd + '/' + file[-24:-4]
            else:
                save_path = dst_dir_path + sd + '/' + file[-23:-4]

            # print(save_path)

            df = pd.read_csv(file, header=0)
            points = df["points"]
            points = np.array(points)
            num = len(points)
            #print(len(points))

            start = 0
            while start + frame_together < len(points):
                fig = plt.figure(figsize=(10, 10))
                # ax1 = fig.add_subplot(2, 2, 1, projection='3d')
                ax2 = fig.add_subplot(2, 2, 2)
                ax3 = fig.add_subplot(2, 2, 3)
                ax4 = fig.add_subplot(2, 2, 4)

                for i in range(frame_together):
                    # for i in range(0, 2):
                    data = eval(points[i + start])
                    length = len(data)
                    if length != 0:
                        zm = func_ReadData(data)
                        #print(zm.shape)
                        # fig = plt.figure(1)

                        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
                        ax1.scatter(zm[:, 1], zm[:, 2], zm[:, 4], marker='.')
                        ax1.set_xlim([0, 5])
                        ax1.set_ylim([-60, 60])
                        ax1.set_zlim([-60, 60])
                        ax1.set_xlabel('Range(m)')
                        ax1.set_ylabel('Azimuth(°)')
                        ax1.set_zlabel('Elevation(°)')

                        # ax2 = fig.add_subplot(2, 2, 2)
                        # print(zm[:, 1])
                        ax2.plot(i*np.ones(length), zm[:, 1], '.')
                        ax2.xaxis.set_major_locator(MultipleLocator(2))
                        ax2.set_xlim([0, 20])
                        ax2.set_ylim([0, 5])
                        ax2.set_xlabel('FrameNum')
                        ax2.set_ylabel('Range(m)')

                        # ax3 = fig.add_subplot(2, 2, 3)
                        ax3.plot(i*np.ones(length), zm[:, 2], '.')
                        ax3.xaxis.set_major_locator(MultipleLocator(2))
                        ax3.set_xlim([0, 20])
                        ax3.set_ylim([-60, 60])
                        ax3.set_xlabel('FrameNum')
                        # ax3.set_ylabel('Azimuth(°)')
                        ax3.set_ylabel('Height')

                        # ax4 = fig.add_subplot(2, 2, 4)
                        ax4.plot(i*np.ones(length), zm[:, 4], '.')
                        ax4.xaxis.set_major_locator(MultipleLocator(2))
                        ax4.set_xlim([0, 20])
                        ax4.set_ylim([-60, 60])
                        ax4.set_xlabel('FrameNum')
                        ax4.set_ylabel('Elevation(°)')

                        plt.pause(0.1)

                plt.tight_layout()
                plt.savefig(save_path + '_' + str(start + 1) + '-' + str(start + frame_together) + '.png')
                plt.close(fig)

                start += sliding
