import matplotlib.pyplot as plt

y = [0.46803542045602783, 0.36402753870744957, 0.26001965020504494, 0, 0.4651665579273136,
     0.9303331161937342, 1.3954996732755413, 1.8606662304749057, 2.3258327894828398,
     2.790999357578837, 3.256165904261368]
x = [i for i in range(len(y))]
plt.plot(x, y)
plt.xlabel('Magnification(Compared to Laplace D4)')
plt.ylabel('Dissimilarity(10\u207b\u2076)')
# plt.title('multi')
plt.show()
