import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

# import matplotlib
# matplotlib.use('TKAgg')

I = [40 for i in range(1111)]
I[500:521] = [60 for k in range(20)]
I_xuan = [80, 300]
I_range = [20, 50, 100, 200]
Iapp = 38
vR = -20
phai = 0.04  # 设置常数
gCa = 4.4
V3 = 2
V4 = 30
ECa = 120
EK = -84
EL = -60
gK = 8
gL = 2  # * 0.1
V1 = -1.2
V2 = 18
CM = 1

dt = 0.1  # 设置步长
seg = 1000  # 设置分割段数
t = 0.0  # 设置初始时间
v = vR  # 设置初始电量
n = 0.05

qts = []  #
times = []  #
# 先定义三个空列表
qt = []  # 用来盛放差分得到的q值
time = []  # 用来盛放时间值
qt.append(v)  # 差分得到的q值列表
time.append(t)  # 时间列表
for ii in range(len(I_xuan)):
    for jj in range(len(I_range)):
        dt = 0.1  # 设置步长
        seg = 1000  # 设置分割段数
        t = 0.0  # 设置初始时间
        v = vR  # 设置初始电量
        n = 0.05
        qt = []  # 用来盛放差分得到的q值
        time = []  # 用来盛放时间值
        qt.append(v)  # 差分得到的q值列表
        time.append(t)  # 时间列表
        I = [38 for i in range(1111)]
        I[500:(500 + I_range[jj] + 1)] = [I_xuan[ii] for k in range(I_range[jj])]
        for i in tqdm(range(seg)):
            t = t + dt
            Iapp = I[i]
            n1 = n + phai * ((0.5 * (1 + math.tanh((v - V3) / V4)) - n) * (math.cosh((v - V3) / (2 * V4)))) * dt
            v1 = (v + (Iapp - gL * (v - EL) - gK * n * (v - EK) - gCa * (0.5 * (1 + math.tanh((v - V1) / V2))) * (
                    v - ECa)) * dt) / CM
            n = n + phai * ((0.5 * (1 + math.tanh((0.5 * (v1 + v) - V3) / V4)) - 0.5 * (n1 + n)) * (
                math.cosh((0.5 * (v1 + v) - V3) / (2 * V4)))) * dt
            v = (v + (Iapp - gL * (0.5 * (v1 + v) - EL) - gK * 0.5 * (n1 + n) * (0.5 * (v1 + v) - EK) - gCa * 0.5 * (
                    1 + math.tanh((0.5 * (v1 + v) - V1) / V2)) * (0.5 * (v1 + v) - ECa)) * dt) / CM  # 差分递推关系
            qt.append(v)  # 差分得到的q值列表
            time.append(t)  # 时间列表
        qts.append(qt)
        times.append(time)

print(qts[0])
plt.figure()
for i in range(len(I_xuan)):
    for j in range(len(I_range)):
        plt.plot(times[i * 4 + j], qts[i * 4 + j], '-', label=f'{I_xuan[i]} - {I_range[j]}')  # 差分得到的电量随时间的变化

plt.xlabel('t')
plt.ylabel('V')
# plt.xlim(0, 99)
plt.ylim(-80, 80)
plt.legend(loc='upper right')
plt.show()
