from sklearn.model_selection import cross_val_score
import ConvRFClassifier
from ConvRFClassifier import ConvRFClassifier
# general imports
import numpy as np
from sklearn.metrics import accuracy_score
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import sklearn

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

class1 = 0
class2 = 2
fraction_of_train_samples = .02

num_train_samples_class_1 = int(np.sum(cifar_train_labels==class1) * fraction_of_train_samples)
num_train_samples_class_2 = int(np.sum(cifar_train_labels==class2) * fraction_of_train_samples)

# get only train images and labels for class 1 and class 2
cifar_ti = np.concatenate([cifar_train_images[cifar_train_labels==class1][:num_train_samples_class_1], cifar_train_images[cifar_train_labels==class2][:num_train_samples_class_2]])
cifar_tl = np.concatenate([np.repeat(0, num_train_samples_class_1), np.repeat(1, num_train_samples_class_2)])

CRFC = ConvRFClassifier()
scores = cross_val_score(CRFC, cifar_ti, cifar_tl, scoring='accuracy', n_jobs = 2)
