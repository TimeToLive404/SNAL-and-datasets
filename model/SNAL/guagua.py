import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


# import matplotlib
# matplotlib.use('TKAgg')


def get_Iapp(file):  # 从保存的结果中提取预测值，保存为列表
    pre_file = open(file)
    lines = pre_file.readlines()
    Iapp = []
    for line in lines:
        datas = line.split(' ')
        Iapp.append(datas[-4].rstrip(','))
    return Iapp


'''

I = [38 for i in range(1111)]
I[500:521] = [60 for k in range(20)]
I_xuan = [30, 60, 80, 100, 200, 300, 800]
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
    I[500:521] = [I_xuan[ii] for k in range(20)]
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

print(qts[3])
plt.figure()
for i in tqdm(range(len(I_xuan))):
    plt.plot(times[i], qts[i], '-', label=f'{I_xuan[i]}')  # 差分得到的电量随时间的变化

    plt.xlabel('t')
    plt.ylabel('V')
    # plt.xlim(0, 99)
    plt.ylim(-80, 80)
    plt.legend(loc='upper right')
plt.show()
'''
# 调制性中间神经元
I = [38 for i in range(1111)]
I[500:521] = [60 for k in range(20)]
I_xuan = [30, 60, 80, 100, 200, 300, 800]
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

# 先定义三个空列表
qt = []  # 用来盛放差分得到的q值
time = []  # 用来盛放时间值
qt.append(v)  # 差分得到的q值列表
time.append(t)  # 时间列表

ress = []  #
times = []  #


def Mineuron(Iapp, num_frame):
    global v, n, t, qt, time, qts, times
    if num_frame % 100 == 0:  # 重新初始化
        t = 0.0  # 设置初始时间
        v = vR  # 设置初始电量
        n = 0.05
        qt = []  # 用来盛放差分得到的q值
        time = []  # 用来盛放时间值

    for i in range(100):
        t = t + dt
        n1 = n + phai * ((0.5 * (1 + math.tanh((v - V3) / V4)) - n) * (math.cosh((v - V3) / (2 * V4)))) * dt
        v1 = (v + (Iapp - gL * (v - EL) - gK * n * (v - EK) - gCa * (0.5 * (1 + math.tanh((v - V1) / V2))) * (
                v - ECa)) * dt) / CM
        n = n + phai * ((0.5 * (1 + math.tanh((0.5 * (v1 + v) - V3) / V4)) - 0.5 * (n1 + n)) * (
            math.cosh((0.5 * (v1 + v) - V3) / (2 * V4)))) * dt
        v = (v + (Iapp - gL * (0.5 * (v1 + v) - EL) - gK * 0.5 * (n1 + n) * (0.5 * (v1 + v) - EK) - gCa * 0.5 * (
                1 + math.tanh((0.5 * (v1 + v) - V1) / V2)) * (0.5 * (v1 + v) - ECa)) * dt) / CM  # 差分递推关系
        qt.append(v)  # 差分得到的q值列表
        time.append(t)  # 时间列表
    # qts.append(qt)
    # times.append(time)
    qt.sort()
    res = qt[-1]
    ress.append(res)

    return res  # 输出qt中最大的v


Iapp_list = get_Iapp('D:/SNAL/result/log_test.txt')  # 得到输入
for i, I in enumerate(Iapp_list):
    res = Mineuron(float(I), i)
    print(res)
times=[i for i in range(200)]
plt.figure()
plt.plot(times, ress, '-o', label=f'555')  # 差分得到的电量随时间的变化
plt.xlabel('t')
plt.ylabel('V')
# plt.xlim(0, 99)
# plt.ylim(-80, 80)
plt.legend(loc='upper right')
plt.show()
