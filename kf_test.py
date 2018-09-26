

import numpy as np
import scipy.io as scio
import psycopg2
import math
import matplotlib.pyplot as plt
# import ppygis
# import datetime
# import string
# import sys
# import logging
# import GeowayLogger
import csv


from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

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

def accuracy(predictions, labels):
    return np.mean(np.sqrt(np.sum((predictions - labels)**2, 1))), np.sqrt(np.sum((predictions - labels)**2, 1))

def kalman_tracker(point_x, point_y):
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    dt = 0.
    # 状态转移矩阵
    tracker.F = np.array([[1, dt, 0,  0],
                          [0,  1, 0,  0],
                          [0,  0, 1, dt],
                          [0,  0, 0,  1]])
    # 用filterpy计算Q矩阵
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.001)
    # tracker.Q = block_diag(q, q)
    weigth = 3
    tracker.Q = np.eye(4) * weigth
    # tracker.B = 0
    # 观测矩阵
    tracker.H = np.array([[1., 0, 0, 0],
                          [0, 0, 1., 0]])
    # R矩阵
    tracker.R = np.array([[4., 0],
                          [0, 4.]])
    # 初始状态和初始P
    tracker.x = np.array([[point_x, 0, point_y, 0]]).T
    tracker.P = np.zeros([4, 4])
    return tracker

# 数据库连接参数
conn = psycopg2.connect(database="gateway", user="postgres", password="postgres", host="192.168.1.248", port="5432")
cur = conn.cursor()


sql = "SELECT c.timestamp, c.uuid, st_asewkt(c.point), st_x(c.point), st_y(c.point), c.direction, c.speed \
FROM t_position c ORDER BY c.timestamp ASC"
cur.execute(sql)
rows = cur.fetchall()  # all rows in table

bs = np.zeros((len(rows), 4, 4))
us = np.zeros((len(rows), 4, 1))
xx = np.zeros((len(rows), 2, 1))


temp = rows[0]
timestamp = rows[0][0]
fx, fy = lonLat2Mercator(rows[0][3], rows[0][4])
tmp_x = fx
tmp_y = fy

for i in range(len(rows)):
    bs[i, :] = np.eye(4) * (rows[i][0] - timestamp)
    timestamp = rows[i][0]
    x, y = lonLat2Mercator(rows[i][3], rows[i][4])
    xx[i, 0, :] = x
    xx[i, 1, :] = y
    us[i, 2, :] = 0 if (rows[i][0] == timestamp) else (x-tmp_x) / (rows[i][0] - timestamp)
    us[i, 0, :] = 0 if (rows[i][0] == timestamp) else (y-tmp_y) / (rows[i][0] - timestamp)
    tmp_x = x
    tmp_y = y


# for i in range(len(rows)):
#     bs[i, :] = np.eye(4) * (rows[i][0] - timestamp)
#     timestamp = rows[i][0]
#     x, y = lonLat2Mercator(rows[i][3], rows[i][4])
#     xx[i, 0, :] = x
#     xx[i, 1, :] = y
#     us[i, 2, :] = math.sin(math.radians(90 - rows[i][5])) * rows[i][6] / 3.6
#     us[i, 0, :] = math.cos(math.radians(90 - rows[i][5])) * rows[i][6] / 3.6

tracker = kalman_tracker(fx, fy)
mu, cov, mup, pre = tracker.batch_filter(xx, Bs=bs, us=us)
kf_predictions = mup[:, [0, 2], :].reshape(len(rows), 2)
kf_xx = xx[:, :, :].reshape(len(rows), 2)
acc, accData = accuracy(kf_predictions, kf_xx)
print("accuracy: ",    acc, "m")

x_i = range(0, len(rows))
tr, = plt.plot(kf_xx[x_i, 0], kf_xx[x_i, 1], 'k-', linewidth=3)
pf, = plt.plot(kf_predictions[x_i, 0], kf_predictions[x_i, 1], 'r-')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()

plt.plot(x_i, accData[x_i], 'b-')
plt.show()

conn.commit()
cur.close()
conn.close()






