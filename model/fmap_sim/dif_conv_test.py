import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import FmapAnalysis, frame_handle

# 不同的卷积核
conv1 = [[0, 1, 0],  # Laplace D4算子
         [1, -4, 1],
         [0, 1, 0]]
conv2 = [[1, 1, 1],  # Laplace D8算子
         [1, -8, 1],
         [1, 1, 1]]
conv3 = [[1, 0, -1],  # sobel算子
         [2, 0, -2],
         [1, 0, -1]]
conv4 = [[1, 2, 1],  # sobel算子
         [0, 0, 0],
         [-1, -2, -1]]
conv5 = [[0, 1, 2],  # sobel算子，提取45°角的filter
         [-1, 0, 1],
         [-2, -1, 0]]
conv6 = [[1, 0, -1],  # prewitt算子
         [1, 0, -1],
         [1, 0, -1]]
conv7 = [[1, 1, 0],  # prewitt算子
         [1, 0, -1],
         [0, -1, -1]]
conv8 = [[-1, -1, -1],  # 锐化卷积核
         [-1, 8, -1],
         [-1, -1, -1]]
conv9 = [[-1, -1, 2],  # 随机定义算子
         [4, -3, -1],
         [1, -1, -1]]
conv10 = [[1, -1, -1],  # 随机定义算子
          [-1, 1, 0],
          [1, -1, 1]]
conv11 = [[0, 0, 0],  # 随机定义算子
          [0, 0, 0],
          [0, 0, 0]]

# 只是模糊图像的卷积核，与上述差异较大
conv12 = [[1, 1, 1],  # 均值算子
          [1, 1, 1],
          [1, 1, 1]]
conv13 = [[1, 2, 1],  # 高斯算子
          [2, 4, 2],
          [1, 2, 1]]


def myconv2D(image, conv, stride=1, kernel_size=3):
    assert image.ndim == 2
    h, w = image.shape
    print('原图形状：', image.shape)
    fmap = []
    for j in tqdm(range(0, h, stride)):
        for i in range(0, w, stride):
            if j + kernel_size <= h and i + kernel_size <= w:
                pixel = np.sum(np.multiply(image[j:j + kernel_size, i:i + kernel_size], conv))
                fmap.append(pixel)
    fmap_np = np.array(fmap)
    fmap_np = fmap_np.reshape(((h - kernel_size) // stride + 1, -1))
    return fmap_np


def myconv2D_new(i, conv_kernel, weight=1):
    i_transformed = np.copy(i)
    size_x = i_transformed.shape[0]
    size_y = i_transformed.shape[1]
    for x in tqdm(range(1, size_x - 1)):
        for y in range(1, size_y - 1):
            convolution = 0
            convolution = convolution + (i[x - 1, y - 1] * conv_kernel[0][0])  # 计算九个像素点与卷积核的卷积
            convolution = convolution + (i[x, y - 1] * conv_kernel[0][1])
            convolution = convolution + (i[x + 1, y - 1] * conv_kernel[0][2])
            convolution = convolution + (i[x - 1, y] * conv_kernel[1][0])
            convolution = convolution + (i[x, y] * conv_kernel[1][1])
            convolution = convolution + (i[x + 1, y] * conv_kernel[1][2])
            convolution = convolution + (i[x - 1, y + 1] * conv_kernel[2][0])
            convolution = convolution + (i[x, y + 1] * conv_kernel[2][1])
            convolution = convolution + (i[x + 1, y + 1] * conv_kernel[2][2])
            convolution = convolution * weight  # ?
            if convolution < 0:
                convolution = 0
            elif convolution > 255:
                convolution = 255
            i_transformed[x, y] = convolution  # 将卷积后的值放入对应位置
    return i_transformed


def trend_multiple(kernel, img, multi=[1, 0.5, 2]):
    plt.figure()
    fmaps = []
    for i, k in enumerate(multi):
        plt.subplot(4, 3, i + 1)
        kernel_now = np.multiply(kernel, k)
        img_np = np.array(img)
        img_np = frame_handle(img_np, 416)
        img_np = np.array(img_np.squeeze())
        fmap = myconv2D_new(img_np, kernel_now)
        # fmap = myconv2D_new(fmap, kernel_now)
        fmaps.append(fmap)
        plt.imshow(fmap, cmap='gray')
    plt.show()
    fmap_base = fmaps.pop(0)
    dis_list = []
    for fmap in fmaps:
        analy = FmapAnalysis([fmap_base], [fmap])
        result = analy.sim_dis()
        dis_list.append(result)
    print(dis_list)


def trend_deep(kernel, img, maxdeep=5):
    plt.figure()
    fmaps = []
    img_np = np.array(img)
    img_np = frame_handle(img_np, 416)
    img_np = np.array(img_np.squeeze())
    for i in range(maxdeep):
        plt.subplot(maxdeep // 3 + 1, 3, i + 1)
        fmap = myconv2D_new(img_np, kernel, i + 1)
        fmaps.append(fmap)
        plt.imshow(fmap, cmap='gray')
        # 将fmap填充为原图大小
        img_np = pad_fmap(fmap)
    plt.show()
    fmap_base = fmaps.pop(0)
    dis_list = []
    for fmap in fmaps:
        analy = FmapAnalysis([fmap_base], [fmap])
        result = analy.sim_dis()
        dis_list.append(result)
    print(dis_list)


def pad_fmap(fmap):
    padded_fmap = np.pad(fmap, ((0, 0), (0, 0)), 'constant', constant_values=(0, 0))
    return padded_fmap


def trend_type(img, kernels=[conv1, conv2]):
    plt.figure()
    fmaps = []
    for i, kernel in enumerate(kernels):
        img_np = np.array(img)
        img_np = frame_handle(img_np, 416)
        img_np = np.array(img_np.squeeze())
        # plt.imshow(img_np)
        plt.subplot(4, 3, i + 1)
        fmap = myconv2D_new(img_np, kernel)
        fmaps.append(fmap)
        plt.imshow(fmap, cmap='gray')
    plt.show()
    fmap_base = fmaps.pop(0)
    dis_list = []
    for fmap in fmaps:
        analy = FmapAnalysis([fmap_base], [fmap])
        result = analy.sim_dis()
        dis_list.append(result)
    print(dis_list)


'''
# 不同倍率下的卷积核相对于1倍自身的相似度变化
img = Image.open('0001.png')
img = img.convert('L')
plt.imshow(img, cmap='gray')
multi = [1, 0.1, 0.3, 0.5, 2, 3, 4, 5, 6, 7, 8]
trend_multiple(conv1, img, multi)
'''

# 不同深度下的同一卷积核[1，-4，1]相对于深度1时的相似度变化
img = Image.open('0001.png')
img = img.convert('L')
plt.imshow(img, cmap='gray')
trend_deep(conv1, img, 53)
''''''
'''
# 不同卷积核深度1时的相似度，相对于[1，-4，1]的那个卷积核
img = Image.open('0001.png')
img = img.convert('L')
plt.imshow(img, cmap='gray')
kernels = [conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11]  #
trend_type(img, kernels)
'''
'''
# 最初测试程序
plt.figure()
plt.subplot(1, 3, 1)
img = Image.open('0001.png')
img = img.convert('L')
plt.imshow(img, cmap='gray')

plt.subplot(1, 3, 2)
img_np = np.array(img)
# img_np = img_np / 255.0
print(img_np)
img_np = frame_handle(img_np, 416)
img_np = np.array(img_np.squeeze())
print(img_np)
fmap1 = myconv2D_new(img_np, conv1)
fmap1 = myconv2D_new(fmap1, conv1)
plt.imshow(fmap1, cmap='gray')
# plt.show()
plt.subplot(1, 3, 3)
fmap2 = myconv2D_new(img_np, conv1_1)
fmap2 = myconv2D_new(fmap2, conv1_1)
# print(fmap2)
plt.imshow(fmap2, cmap='gray')
plt.show()

analy = FmapAnalysis([fmap1], [fmap2])
result = analy.sim_dis()
print(result)
'''
