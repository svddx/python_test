

import numpy as np
import scipy.io as scio
import psycopg2
import math
import matplotlib.pyplot as plt
import pandas as pd
# import ppygis
# import datetime
# import string
# import sys
# import logging
# import GeowayLogger
import csv


def lonLat2Mercator(lat, lng):
    x = lng * 20037508.342789/180
    y = math.log(math.tan((90+lat)*math.pi/360))/(math.pi/180)
    y = y * 20037508.34789/180
    return x, y

def Mercator2lonLat(mercatorX, mercatorY):

    lng = mercatorX/20037508.34*180
    lat = mercatorY/20037508.34*180
    lat = 180/math.pi*(2*math.atan(math.exp(lat * math.pi/180)) - math.pi/2)
    return lng, lat

CSV_COLUMN_NAMES = ['count', 'uuid',
                    'gp_pt', 'lat', 'lng']

csv_data = pd.read_csv('solution_1_b0487a260994.csv', header=None)  # 读取数据
print(csv_data.shape)  # (189, 9)

# N = 5
# csv_batch_data = csv_data.tail(N)  # 取后5条数据
# print(csv_batch_data.shape)  # (5, 9)
# train_batch_data = csv_batch_data[list(range(3, 6))]  # 取这20条数据的3到5列值(索引从0开始)
# print(train_batch_data)



# rows = cur.fetchall()  # all rows in table
#
# bs = np.zeros((len(rows), 4, 4))
# us = np.zeros((len(rows), 4, 1))
xx = np.zeros((len(csv_data), 2))
#
#
# temp = rows[0]
# timestamp = rows[0][0]
# fx, fy = lonLat2Mercator(rows[0][3], rows[0][4])
# tmp_x = fx
# tmp_y = fy
#
for i in range(len(csv_data)):
    x, y = lonLat2Mercator(csv_data[3][i], csv_data[4][i])
    xx[i, 0] = x
    xx[i, 1] = y

#
#

plt.scatter(xx[:, 1], xx[:, 0], marker='x', color='g', label='1', s=30)

# x_i = range(0, len(csv_data))
# tr, = plt.plot(xx[x_i, 0], xx[x_i, 1], '*')
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
plt.show()







