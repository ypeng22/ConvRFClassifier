import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

#convrf vs naiverf
kappas_conv_naive = []
xs = []
for class1 in range(10):
    for class2 in range(class1 + 1, 10):
        df = pd.read_csv("cifar_results/" + str(class1) + "_vs_" + str(class2) + ".csv", index_col=0)
        rferr = .5 / (1 - df.iloc[0])
        convrferr = .5 / (1 - df.iloc[1])
        kappa = convrferr - rferr
        kappas_conv_naive.extend(kappa)
        xs.extend([1, 2, 3, 4, 5, 6, 7, 8])

ind1 = range(0, len(xs), 8)
ind2 = range(1, len(xs), 8)
ind3 = range(2, len(xs), 8)
ind4 = range(3, len(xs), 8)
ind5 = range(4, len(xs), 8)
ind6 = range(5, len(xs), 8)
ind7 = range(6, len(xs), 8)
ind8 = range(7, len(xs), 8)

plt.clf()
plt.scatter(xs, kappas_conv_naive)
plt.xticks([1,2, 3, 4, 5, 6, 7, 8], ['50', '97', '186', '360', '695', '1341', '2590', '5000'])
plt.xlabel('Number of Train Samples')
plt.ylabel('Kappa')
plt.title('Convrf kappa - NaiveRF kappa vs # of Train Samples')
plt.savefig('kappas/convrf-naiverf kappa scatter')

plt.clf()
box1 = [(kappas_conv_naive[i]) for i in ind1]
box2 = [(kappas_conv_naive[i]) for i in ind2]
box3 = [(kappas_conv_naive[i]) for i in ind3]
box4 = [(kappas_conv_naive[i]) for i in ind4]
box5 = [(kappas_conv_naive[i]) for i in ind5]
box6 = [(kappas_conv_naive[i]) for i in ind6]
box7 = [(kappas_conv_naive[i]) for i in ind7]
box8 = [(kappas_conv_naive[i]) for i in ind8]
boxs = [box1, box2, box3, box4, box5, box5, box7, box8]

plt.boxplot(boxs)
plt.xlabel('Number of Train Samples')
plt.ylabel('Kappa')
plt.title('Convrf kappa - NaiveRF kappa vs # of Train Samples')
plt.xticks([1,2, 3, 4, 5, 6, 7, 8], ['50', '97', '186', '360', '695', '1341', '2590', '5000'])
plt.savefig('kappas/convrf-naiverf kappa box')




kappas_conv_cnn = []
xs = []
for class1 in range(10):
    for class2 in range(class1 + 1, 10):
        df = pd.read_csv("cifar_results/" + str(class1) + "_vs_" + str(class2) + ".csv", index_col=0)
        rferr = .5 / (1 - df.iloc[3])
        convrferr = .5 / (1 - df.iloc[1])
        kappa = convrferr - rferr
        kappas_conv_cnn.extend(kappa)
        xs.extend([1, 2, 3, 4, 5, 6, 7, 8])

ind1 = range(0, len(xs), 8)
ind2 = range(1, len(xs), 8)
ind3 = range(2, len(xs), 8)
ind4 = range(3, len(xs), 8)
ind5 = range(4, len(xs), 8)
ind6 = range(5, len(xs), 8)
ind7 = range(6, len(xs), 8)
ind8 = range(7, len(xs), 8)

plt.clf()
plt.scatter(xs, kappas_conv_cnn)
plt.xticks([1,2, 3, 4, 5, 6, 7, 8], ['50', '97', '186', '360', '695', '1341', '2590', '5000'])
plt.xlabel('Number of Train Samples')
plt.ylabel('Kappa')
plt.title('Convrf kappa - SimpleCNN kappa vs # of Train Samples')
plt.savefig('kappas/convrf-SimpleCNN kappa scatter')
plt.clf()
box1 = [(kappas_conv_cnn[i]) for i in ind1]
box2 = [(kappas_conv_cnn[i]) for i in ind2]
box3 = [(kappas_conv_cnn[i]) for i in ind3]
box4 = [(kappas_conv_cnn[i]) for i in ind4]
box5 = [(kappas_conv_cnn[i]) for i in ind5]
box6 = [(kappas_conv_cnn[i]) for i in ind6]
box7 = [(kappas_conv_cnn[i]) for i in ind7]
box8 = [(kappas_conv_cnn[i]) for i in ind8]
boxs = [box1, box2, box3, box4, box5, box5, box7, box8]

plt.boxplot(boxs)
plt.xlabel('Number of Train Samples')
plt.ylabel('Kappa')
plt.title('Convrf kappa - SimpleCNN kappa vs # of Train Samples')
plt.xticks([1,2, 3, 4, 5, 6, 7, 8], ['50', '97', '186', '360', '695', '1341', '2590', '5000'])
plt.savefig('kappas/convrf-SimpleCNN kappa box')


kappas_conv_cnn32 = []
xs = []
for class1 in range(10):
    for class2 in range(class1 + 1, 10):
        df = pd.read_csv("cifar_results/" + str(class1) + "_vs_" + str(class2) + ".csv", index_col=0)
        rferr = .5 / (1 - df.iloc[5])
        convrferr = .5 / (1 - df.iloc[1])
        kappa = convrferr - rferr
        kappas_conv_cnn32.extend(kappa)
        xs.extend([1, 2, 3, 4, 5, 6, 7, 8])

ind1 = range(0, len(xs), 8)
ind2 = range(1, len(xs), 8)
ind3 = range(2, len(xs), 8)
ind4 = range(3, len(xs), 8)
ind5 = range(4, len(xs), 8)
ind6 = range(5, len(xs), 8)
ind7 = range(6, len(xs), 8)
ind8 = range(7, len(xs), 8)

plt.clf()
plt.scatter(xs, kappas_conv_cnn32)
plt.xticks([1,2, 3, 4, 5, 6, 7, 8], ['50', '97', '186', '360', '695', '1341', '2590', '5000'])
plt.xlabel('Number of Train Samples')
plt.ylabel('Kappa')
plt.title('Convrf kappa - CNN32_2 kappa vs # of Train Samples')
plt.savefig('kappas/convrf-CNN32_2 kappa scatter')
plt.clf()
box1 = [(kappas_conv_cnn32[i]) for i in ind1]
box2 = [(kappas_conv_cnn32[i]) for i in ind2]
box3 = [(kappas_conv_cnn32[i]) for i in ind3]
box4 = [(kappas_conv_cnn32[i]) for i in ind4]
box5 = [(kappas_conv_cnn32[i]) for i in ind5]
box6 = [(kappas_conv_cnn32[i]) for i in ind6]
box7 = [(kappas_conv_cnn32[i]) for i in ind7]
box8 = [(kappas_conv_cnn32[i]) for i in ind8]
boxs = [box1, box2, box3, box4, box5, box5, box7, box8]

plt.boxplot(boxs)
plt.xlabel('Number of Train Samples')
plt.ylabel('Kappa')
plt.title('Convrf kappa - CNN32_2 kappa vs # of Train Samples')
plt.xticks([1,2, 3, 4, 5, 6, 7, 8], ['50', '97', '186', '360', '695', '1341', '2590', '5000'])
plt.savefig('kappas/convrf-CNN32_2 kappa box')