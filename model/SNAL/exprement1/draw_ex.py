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
        ex.append(datas[-1])
    return ex
def get_res2(file):
    file_con = open(file, 'r')
    lines = file_con.readlines()
    ex = []
    for line in lines:
        datas = line.split(' ')
        ex.append(datas[-2][1])
    return ex

plt.figure()
plt.subplot(1, 3, 1)
label = [1 for _ in range(199)]
label[0:21] = [0 for _ in range(21)]
label[115:138] = [0 for _ in range(23)]
label[197:199] = [0 for _ in range(3)]
times = [i for i in range(200)]
plt.plot(times, label, '-', label=f'Modulated interneuron output')  # 差分得到的电量随时间的变化
plt.subplot(1, 3, 2)
label = [1 for _ in range(199)]
label[0:21] = [0 for _ in range(21)]
label[115:138] = [0 for _ in range(23)]
label[197:199] = [0 for _ in range(3)]
times = [i for i in range(200)]
plt.plot(times, label, '-', label=f'Modulated interneuron output')  # 差分得到的电量随时间的变化
plt.subplot(1, 3, 3)
label = [1 for _ in range(199)]
label[0:21] = [0 for _ in range(21)]
label[115:138] = [0 for _ in range(23)]
label[197:199] = [0 for _ in range(3)]
times = [i for i in range(200)]
plt.plot(times, label, '-', label=f'Modulated interneuron output')  # 差分得到的电量随时间的变化

plt.subplot(1, 3, 1)
ress = get_res2('D:/SNAL/result/log_tttx1.txt')
ress=[float(r) for r in ress]
times = [i for i in range(200)]
plt.plot(times, ress, '-', label=f'Modulated interneuron output')  # 差分得到的电量随时间的变化
plt.xlabel('t')
plt.ylabel('V')
# plt.xlim(0, 99)
# plt.ylim(-80, 80)
# plt.legend(loc='upper right')
plt.subplot(1, 3, 2)
ress = get_res2('D:/SNAL/result/log_tttx22.txt')
ress=[float(r) for r in ress]
times = [i for i in range(200)]
plt.plot(times, ress, '-', label=f'Modulated interneuron output')  # 差分得到的电量随时间的变化
plt.xlabel('t')
plt.ylabel('V')
# plt.xlim(0, 99)
# plt.ylim(-80, 80)
# plt.legend(loc='upper right')
plt.subplot(1, 3, 3)
ress = get_res2('D:/SNAL/result/log_tttx3.txt')
ress=[float(r) for r in ress]
times = [i for i in range(200)]
plt.plot(times, ress, '-', label=f'Modulated interneuron output')  # 差分得到的电量随时间的变化
plt.xlabel('t')
plt.ylabel('V')
# plt.xlim(0, 99)
# plt.ylim(-80, 80)
# plt.legend(loc='upper right')


plt.subplot(1, 3, 1)
ress = get_res('D:/SNAL/result/log_exx.txt')
ress=[float(r) for r in ress]
times = [i for i in range(200)]
plt.plot(times, ress, '-o', label=f'Modulated interneuron output')  # 差分得到的电量随时间的变化
plt.xlabel('t')
plt.ylabel('V')
# plt.xlim(0, 99)
# plt.ylim(-80, 80)
# plt.legend(loc='upper right')

plt.subplot(1, 3, 2)
ress = get_res('D:/SNAL/result/log_exx20000.txt')
ress=[float(r) for r in ress]
times = [i for i in range(200)]
plt.plot(times, ress, '-o', label=f'Modulated interneuron output')  # 差分得到的电量随时间的变化
plt.xlabel('t')
plt.ylabel('V')
# plt.xlim(0, 99)
# plt.ylim(-80, 80)
# plt.legend(loc='upper right')

plt.subplot(1, 3, 3)
ress = get_res('D:/SNAL/result/log_exx200000.txt')
ress=[float(r) for r in ress]
times = [i for i in range(200)]
plt.plot(times, ress, '-o', label=f'Modulated interneuron output')  # 差分得到的电量随时间的变化
plt.xlabel('t')
plt.ylabel('V')
# plt.xlim(0, 99)
# plt.ylim(-80, 80)
# plt.legend(loc='upper right')

plt.show()
