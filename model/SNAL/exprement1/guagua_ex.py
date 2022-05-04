import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


def get_res(file):
    file_con = open(file, 'r')
    lines = file_con.readlines()
    ex = []
    for line in lines:
        datas = line.split(' ')
        ex.append(datas[-2].rstrip(','))
    return ex

plt.figure()
plt.subplot(2, 3, 1)
ress = get_res('D:/SNAL/result/log_ttt1.txt')
ress=[float(r) for r in ress]
times = [i for i in range(200)]
plt.plot(times, ress, '-o', label=f'Modulated interneuron output')  # 差分得到的电量随时间的变化
plt.xlabel('t')
plt.ylabel('V')
# plt.xlim(0, 99)
# plt.ylim(-80, 80)
# plt.legend(loc='upper right')

plt.subplot(2, 3, 2)
ress = get_res('D:/SNAL/result/log_ttt2.txt')
ress=[float(r) for r in ress]
times = [i for i in range(200)]
plt.plot(times, ress, '-o', label=f'Modulated interneuron output')  # 差分得到的电量随时间的变化
plt.xlabel('t')
plt.ylabel('V')
# plt.xlim(0, 99)
# plt.ylim(-80, 80)
# plt.legend(loc='upper right')

plt.subplot(2, 3, 3)
ress = get_res('D:/SNAL/result/log_ttt4.txt')
ress=[float(r) for r in ress]
times = [i for i in range(200)]
plt.plot(times, ress, '-o', label=f'Modulated interneuron output')  # 差分得到的电量随时间的变化
plt.xlabel('t')
plt.ylabel('V')
# plt.xlim(0, 99)
# plt.ylim(-80, 80)
# plt.legend(loc='upper right')
plt.show()