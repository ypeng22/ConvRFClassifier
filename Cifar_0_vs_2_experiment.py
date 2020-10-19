# general imports
import random
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.utils.data as utils
from torch.autograd import Variable
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns; sns.set()
import ConvRFClassifier


plt.rcParams["legend.loc"] = "best"
plt.rcParams['figure.facecolor'] = 'white'
#matplotlib inline
# filter python warnings
import warnings
warnings.filterwarnings("ignore")
# prepare CIFAR data

# normalize
scale = np.mean(np.arange(0, 256))
normalize = lambda x: (x - scale) / scale

# train data
cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
cifar_train_images = normalize(cifar_trainset.data)
cifar_train_labels = np.array(cifar_trainset.targets)

# test data
cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
cifar_test_images = normalize(cifar_testset.data)
cifar_test_labels = np.array(cifar_testset.targets)

print(np.min(cifar_train_images))
print(np.max(cifar_train_images))

class1 = 0
class2 = 2
fraction_of_train_samples = .002

num_train_samples_class_1 = int(np.sum(cifar_train_labels==class1) * fraction_of_train_samples)
num_train_samples_class_2 = int(np.sum(cifar_train_labels==class2) * fraction_of_train_samples)

# get only train images and labels for class 1 and class 2
cifar_ti = np.concatenate([cifar_train_images[cifar_train_labels==class1][:num_train_samples_class_1], cifar_train_images[cifar_train_labels==class2][:num_train_samples_class_2]])
cifar_tl = np.concatenate([np.repeat(0, num_train_samples_class_1), np.repeat(1, num_train_samples_class_2)])

# get only test images and labels for class 1 and class 2
cifar_testi = np.concatenate([cifar_test_images[cifar_test_labels==class1], cifar_test_images[cifar_test_labels==class2]])
cifar_testl = np.concatenate([np.repeat(0, np.sum(cifar_test_labels==class1)), np.repeat(1, np.sum(cifar_test_labels==class2))])
 
 
CRFC = ConvRFClassifier.ConvRFClassifier()

CRFC.fit(cifar_ti, cifar_tl)
predict = CRFC.predict(cifar_testi)
print(accuracy_score(cifar_testl, predict))
