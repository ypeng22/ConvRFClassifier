# general imports
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import ConvRFClassifier
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns; sns.set()
plt.rcParams["legend.loc"] = "best"
plt.rcParams['figure.facecolor'] = 'white'
#%matplotlib inline


# filter python warnings
import warnings
warnings.filterwarnings("ignore")

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()
                
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

# transform
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform) 
    
# define a simple CNN arhcitecture
class SimpleCNNOneFilter(torch.nn.Module):
    
    def __init__(self):
        super(SimpleCNNOneFilter, self).__init__()        
        self.conv1 = torch.nn.Conv2d(3, 1, kernel_size=10, stride=2)
        self.fc1 = torch.nn.Linear(144, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 144)
        x = self.fc1(x)
        return(x)

class SimpleCNN32Filter(torch.nn.Module):
    
    def __init__(self):
        super(SimpleCNN32Filter, self).__init__()        
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=10, stride=2) # try 64 too, if possible
        self.fc1 = torch.nn.Linear(144*32, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 144*32)
        x = self.fc1(x)
        return(x)

class SimpleCNN32Filter2Layers(torch.nn.Module):
    
    def __init__(self):
        super(SimpleCNN32Filter2Layers, self).__init__()        
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=10, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=7, stride=1)
        self.fc1 = torch.nn.Linear(36*32, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 36*32)
        x = self.fc1(x)
        return(x)
    
    
def run_naive_rf(train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1=3, class2=5):
    num_train_samples_class_1 = int(np.sum(train_labels==class1) * fraction_of_train_samples)
    num_train_samples_class_2 = int(np.sum(train_labels==class2) * fraction_of_train_samples)
    
    # get only train images and labels for class 1 and class 2
    train_images = np.concatenate([train_images[train_labels==class1][:num_train_samples_class_1], train_images[train_labels==class2][:num_train_samples_class_2]])
    train_labels = np.concatenate([np.repeat(0, num_train_samples_class_1), np.repeat(1, num_train_samples_class_2)])

    # get only test images and labels for class 1 and class 2
    test_images = np.concatenate([test_images[test_labels==class1], test_images[test_labels==class2]])
    test_labels = np.concatenate([np.repeat(0, np.sum(test_labels==class1)), np.repeat(1, np.sum(test_labels==class2))])

    # Train
    clf = RandomForestClassifier(n_estimators=100, n_jobs = -1)
    clf.fit(train_images.reshape(-1, 32*32*3), train_labels)
    # Test
    test_preds = clf.predict(test_images.reshape(-1, 32*32*3))
    return accuracy_score(test_labels, test_preds)

def run_own_one_layer(train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1=3, class2=5):
    num_train_samples_class_1 = int(np.sum(train_labels==class1) * fraction_of_train_samples)
    num_train_samples_class_2 = int(np.sum(train_labels==class2) * fraction_of_train_samples)
    
    # get only train images and labels for class 1 and class 2
    train_images = np.concatenate([train_images[train_labels==class1][:num_train_samples_class_1], train_images[train_labels==class2][:num_train_samples_class_2]])
    train_labels = np.concatenate([np.repeat(0, num_train_samples_class_1), np.repeat(1, num_train_samples_class_2)])

    # get only test images and labels for class 1 and class 2
    test_images = np.concatenate([test_images[test_labels==class1], test_images[test_labels==class2]])
    test_labels = np.concatenate([np.repeat(0, np.sum(test_labels==class1)), np.repeat(1, np.sum(test_labels==class2))])
    
    ## Train
    conv1 = ConvRFClassifier.ConvRFClassifier(layers = 1, kernel_size = (10,), stride = (2,))
    conv1.fit(train_images, train_labels)

    #test
    mnist_test_preds = conv1.predict(test_images)

    return accuracy_score(test_labels, mnist_test_preds)
   
def run_own_two_layer(train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1=3, class2=5):
    num_train_samples_class_1 = int(np.sum(train_labels==class1) * fraction_of_train_samples)
    num_train_samples_class_2 = int(np.sum(train_labels==class2) * fraction_of_train_samples)
    
    # get only train images and labels for class 1 and class 2
    train_images = np.concatenate([train_images[train_labels==class1][:num_train_samples_class_1], train_images[train_labels==class2][:num_train_samples_class_2]])
    train_labels = np.concatenate([np.repeat(0, num_train_samples_class_1), np.repeat(1, num_train_samples_class_2)])

    # get only test images and labels for class 1 and class 2
    test_images = np.concatenate([test_images[test_labels==class1], test_images[test_labels==class2]])
    test_labels = np.concatenate([np.repeat(0, np.sum(test_labels==class1)), np.repeat(1, np.sum(test_labels==class2))])
    
    ## Train
    conv2 = ConvRFClassifier.ConvRFClassifier(layers = 2, kernel_size = (10,7), stride = (2,1))
    conv2.fit(train_images, train_labels)

    #test
    mnist_test_preds = conv2.predict(test_images)

    return accuracy_score(test_labels, mnist_test_preds)


def cnn_train_test(cnn_model, y_train, y_test, fraction_of_train_samples, class1=3, class2=5):
    # set params
    num_epochs = 5
    learning_rate = 0.001

    class1_indices = np.argwhere(y_train==class1).flatten()
    class1_indices = class1_indices[:int(len(class1_indices) * fraction_of_train_samples)]
    class2_indices = np.argwhere(y_train==class2).flatten()
    class2_indices = class2_indices[:int(len(class2_indices) * fraction_of_train_samples)]
    train_indices = np.concatenate([class1_indices, class2_indices])

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, num_workers=2, sampler=train_sampler)

    test_indices = np.concatenate([np.argwhere(y_test==class1).flatten(), np.argwhere(y_test==class2).flatten()])
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=2, sampler=test_sampler)

    # define model
    net = cnn_model()
    dev = torch.device("cuda:0")
    net.to(dev)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs = torch.tensor(inputs).to(dev)
            labels = torch.tensor(labels).to(dev)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # test the model
    correct = torch.tensor(0).to(dev)
    total = torch.tensor(0).to(dev)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            labels = torch.tensor(labels).to(dev)
            images = torch.tensor(images).to(dev)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.view(-1)).sum().item()
    accuracy = float(correct) / float(total)
    return accuracy


def run_cnn(cnn_model, train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1=3, class2=5):
    return cnn_train_test(cnn_model, train_labels, test_labels, fraction_of_train_samples, class1, class2)

 

   # accuracy vs num training samples (naive_rf)
two_layer_convrf = list()
fraction_of_train_samples_space = np.geomspace(0.01, 1.0, num=10)
for fraction_of_train_samples in fraction_of_train_samples_space:
    best_accuracy = np.mean([run_own_two_layer(cifar_train_images, cifar_train_labels, cifar_test_images, cifar_test_labels, fraction_of_train_samples, 3, 5) for _ in range(2)])
    two_layer_convrf.append(best_accuracy)
    print("Train Fraction:", str(fraction_of_train_samples))
    print("Accuracy:", str(best_accuracy))
    
# accuracy vs num training samples (one layer cnn)
cnn_acc_vs_n = list()
fraction_of_train_samples_space = np.geomspace(0.01, 1.0, num=10)
for fraction_of_train_samples in fraction_of_train_samples_space:
    best_accuracy = np.mean([run_cnn(SimpleCNNOneFilter, cifar_train_images, cifar_train_labels, cifar_test_images, cifar_test_labels, fraction_of_train_samples, 3, 5) for _ in range(2)])
    cnn_acc_vs_n.append(best_accuracy)
    print("Train Fraction:", str(fraction_of_train_samples))
    print("Accuracy:", str(best_accuracy))
   
    
   
# accuracy vs num training samples (naive_rf)
naive_rf_acc_vs_n = list()
fraction_of_train_samples_space = np.geomspace(0.01, 1.0, num=10)
for fraction_of_train_samples in fraction_of_train_samples_space:
    best_accuracy = np.mean([run_naive_rf(cifar_train_images, cifar_train_labels, cifar_test_images, cifar_test_labels, fraction_of_train_samples, 3, 5) for _ in range(2)])
    naive_rf_acc_vs_n.append(best_accuracy)
    print("Train Fraction:", str(fraction_of_train_samples))
    print("Accuracy:", str(best_accuracy))
    
# accuracy vs num training samples (naive_rf)
one_layer_convrf = list()
fraction_of_train_samples_space = np.geomspace(0.01, 1.0, num=10)
for fraction_of_train_samples in fraction_of_train_samples_space:
    best_accuracy = np.mean([run_own_one_layer(cifar_train_images, cifar_train_labels, cifar_test_images, cifar_test_labels, fraction_of_train_samples, 3, 5) for _ in range(2)])
    one_layer_convrf.append(best_accuracy)
    print("Train Fraction:", str(fraction_of_train_samples))
    print("Accuracy:", str(best_accuracy))
    
 
# accuracy vs num training samples (one layer cnn (32 filters))
cnn32_acc_vs_n = list()
fraction_of_train_samples_space = np.geomspace(0.01, 1.0, num=10)
for fraction_of_train_samples in fraction_of_train_samples_space:
    best_accuracy = np.mean([run_cnn(SimpleCNN32Filter, cifar_train_images, cifar_train_labels, cifar_test_images, cifar_test_labels, fraction_of_train_samples, 3, 5) for _ in range(2)])
    cnn32_acc_vs_n.append(best_accuracy)
    print("Train Fraction:", str(fraction_of_train_samples))
    print("Accuracy:", str(best_accuracy))
    

# accuracy vs num training samples (two layer cnn (32 filters))
cnn32_two_layer_acc_vs_n = list()
fraction_of_train_samples_space = np.geomspace(0.01, 1.0, num=10)
for fraction_of_train_samples in fraction_of_train_samples_space:
    best_accuracy = np.mean([run_cnn(SimpleCNN32Filter2Layers, cifar_train_images, cifar_train_labels, cifar_test_images, cifar_test_labels, fraction_of_train_samples, 3, 5) for _ in range(3)])
    cnn32_two_layer_acc_vs_n.append(best_accuracy)
    print("Train Fraction:", str(fraction_of_train_samples))
    print("Accuracy:", str(best_accuracy))
    
    
plt.rcParams['figure.figsize'] = 13, 10
plt.rcParams['font.size'] = 25
plt.rcParams['legend.fontsize'] = 16.5
plt.rcParams['legend.handlelength'] = 2.5
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

fig, ax = plt.subplots() # create a new figure with a default 111 subplot
ax.plot(fraction_of_train_samples_space*100, naive_rf_acc_vs_n, marker='X', markerfacecolor='red', markersize=10, color='green', linewidth=3, linestyle=":", label="Naive RF")
ax.plot(fraction_of_train_samples_space*100, one_layer_convrf, marker='X', markerfacecolor='red', markersize=10, color='green', linewidth=3, linestyle="--", label="Deep Conv RF")
ax.plot(fraction_of_train_samples_space*100, two_layer_convrf, marker='X', markerfacecolor='red', markersize=10, color='green', linewidth=3, label="Deep Conv RF Two Layer")

ax.plot(fraction_of_train_samples_space*100, cnn_acc_vs_n, marker='X', markerfacecolor='red', markersize=10, color='orange', linewidth=3, linestyle=":", label="CNN")
ax.plot(fraction_of_train_samples_space*100, cnn32_acc_vs_n, marker='X', markerfacecolor='red', markersize=10, color='orange', linewidth=3, linestyle="--", label="CNN (32 filters)")
ax.plot(fraction_of_train_samples_space*100, cnn32_two_layer_acc_vs_n, marker='X', markerfacecolor='red', markersize=10, color='orange', linewidth=3, label="CNN Two Layer (32 filters)")

ax.set_xlabel('Percentage of Train Samples', fontsize=18)
ax.set_xscale('log')
ax.set_xticks([i*100 for i in list(np.geomspace(0.01, 1.0, num=10))])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.set_ylabel('Accuracy', fontsize=18)
# ax.set_ylim(0.68, 1)

ax.set_title("3 (cat) vs 5 (dog) Classification", fontsize=18)
plt.legend()
plt.show()

