# general imports
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import ConvRFClassifier
import ConvRFClassifier_predict
import ConvRFClassifier_predict_proba
import matplotlib.pyplot as plt
import matplotlib.image as mpimp
import matplotlib
import seaborn as sns; sns.set()
plt.rcParams["legend.loc"] = "best"
plt.rcParams['figure.facecolor'] = 'white'
#%matplotlib inline

class1 = 3
class2 = 9
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

dev = torch.device('cuda:0')
fraction_of_train_samples_space = np.geomspace(0.01, .2, num=3)



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
    
    def fit(self, y_train, labels):
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
    
        

        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
    
        for epoch in range(num_epochs):  # loop over the dataset multiple times
    
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                inputs, labels = data
                inputs = torch.tensor(inputs).to(dev)
                labels = torch.tensor(labels).to(dev)
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                
    def predict(self, placeholder):
        test_indices = np.concatenate([np.argwhere(cifar_test_labels==class1).flatten(), np.argwhere(cifar_test_labels==class2).flatten()])
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)
    
        test_loader = torch.utils.data.DataLoader(testset, batch_size=16,
                                                 shuffle=False, num_workers=2, sampler=test_sampler)
        with torch.no_grad():
            predicted = torch.tensor([]).to(dev)
            for data in test_loader:
                images, labels = data
                labels = torch.tensor(labels).to(dev)
                images = torch.tensor(images).to(dev)
                outputs = self(images)
                _, batch_pred = torch.max(outputs.data, 1)
                batch_pred[batch_pred != 1] = 0
                predicted = torch.cat((predicted, batch_pred))
            return predicted.cpu().numpy()
        

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
'''   
train_images = cifar_train_images
train_labels = cifar_train_labels
test_images = cifar_test_images
test_labels = cifar_test_labels
model2 = cnn
model = conv_rf
'''   
def run_rf(model, model2, train_images, train_labels, test_images, test_labels, fraction_of_train_samples):
    num_train_samples_class_1 = int(np.sum(train_labels==class1) * fraction_of_train_samples)
    num_train_samples_class_2 = int(np.sum(train_labels==class2) * fraction_of_train_samples)
    
    # get only train images and labels for class 1 and class 2
    train_images = np.concatenate([train_images[train_labels==class1][:num_train_samples_class_1], train_images[train_labels==class2][:num_train_samples_class_2]])
    train_labels = np.concatenate([np.repeat(0, num_train_samples_class_1), np.repeat(1, num_train_samples_class_2)])

    # get only test images and labels for class 1 and class 2
    test_images = np.concatenate([test_images[test_labels==class1], test_images[test_labels==class2]])
    test_labels = np.concatenate([np.repeat(0, np.sum(test_labels==class1)), np.repeat(1, np.sum(test_labels==class2))])

    #fit and predict first model
    modelpred = None
    if isinstance(model, sklearn.ensemble.RandomForestClassifier):
        #print(train_images.shape)
        #print(train_labels.shape)
        model.fit(train_images.reshape(-1, 32*32*3), train_labels)
        modelpred = model.predict(test_images.reshape(-1, 32*32*3))
    
    #fit and predict second model
    model2.fit(train_images, train_labels)
    model2pred = model2.predict(test_images).astype(int)
    #print(type(model2pred))
    #print(model2pred[:10])
    #print(modelpred[:10])
    
    #calculate kohens cappa according to https://en.wikipedia.org/wiki/Cohen%27s_kappa
    #model is A, model2 is B, letters are corresponding to grid, class1 is no, class2 is yes
    a = len(np.where((modelpred == 1) & (model2pred == 1))[0])
    b = len(np.where((modelpred == 1) & (model2pred == 0))[0])
    c = len(np.where((modelpred == 0) & (model2pred == 1))[0])
    d = len(np.where((modelpred == 0) & (model2pred == 0))[0])
    
    s = (a + b + c + d)
    po = (a + d) / s
    pe = (a + b) / s * (a + c) / s + (c + d) / s * (b + d) / s
    return (po - pe) / (1 - pe)

'''
   # accuracy vs num training samples (naive_rf)
conv_vs_simplecnn = list()
for fraction_of_train_samples in fraction_of_train_samples_space:
    #conv_rf = ConvRFClassifier.ConvRFClassifier(layers = 1, kernel_size = (10,), stride = (2,))
    conv_rf = RandomForestClassifier(n_estimators=100, n_jobs = -1)
    cnn = SimpleCNNOneFilter().cuda()
       
    kappa = np.mean([run_rf(conv_rf, cnn, cifar_train_images, cifar_train_labels, cifar_test_images, cifar_test_labels, fraction_of_train_samples) for _ in range(2)])
    conv_vs_simplecnn.append(kappa)
    print("simplecnn vs conv Train Fraction: %.4f" % fraction_of_train_samples, " kappa: %.4f" % kappa)
'''  
    

# accuracy vs num training samples (naive_rf)
kappa_naive_vs_conv = list()
for fraction_of_train_samples in fraction_of_train_samples_space:
    RF = RandomForestClassifier(n_estimators=100, n_jobs = -1)
    conv_rf = ConvRFClassifier.ConvRFClassifier(layers = 1, kernel_size = (10,), stride = (2,))
    
    kappa = np.mean([run_rf(RF, conv_rf, cifar_train_images, cifar_train_labels, cifar_test_images, cifar_test_labels, fraction_of_train_samples) for _ in range(2)])
    kappa_naive_vs_conv.append(kappa)
    print("kappa naive vs conv Train Fraction:", str(fraction_of_train_samples), " kappa:", str(kappa))
  
  
'''# accuracy vs num training samples (one layer cnn)
naive_vs_simplecnn = list()
for fraction_of_train_samples in fraction_of_train_samples_space:
    conv_rf_p = ConvRFClassifier_predict.ConvRFClassifier(layers = 1, kernel_size = (10,), stride = (2,))
    best_accuracy = np.mean([run_rf(conv_rf_p, cifar_train_images, cifar_train_labels, cifar_test_images, cifar_test_labels, fraction_of_train_samples) for _ in range(2)])
    conv_rf_predict.append(best_accuracy)
    print("Train Fraction:", str(fraction_of_train_samples))
    print("Accuracy:", str(best_accuracy))
   

# accuracy vs num training samples (naive_rf)
conv_rf_apply = list()
for fraction_of_train_samples in fraction_of_train_samples_space:
    conv_rf_a = ConvRFClassifier.ConvRFClassifier(layers = 1, kernel_size = (10,), stride = (2,))
    best_accuracy = np.mean([run_rf(conv_rf_a, cifar_train_images, cifar_train_labels, cifar_test_images, cifar_test_labels, fraction_of_train_samples) for _ in range(2)])
    conv_rf_apply.append(best_accuracy)
    print("Train Fraction:", str(fraction_of_train_samples))
    print("Accuracy:", str(best_accuracy))

    

# accuracy vs num training samples (one layer cnn (32 filters))
cnn32_acc_vs_n = list()
for fraction_of_train_samples in fraction_of_train_samples_space:
    best_accuracy = np.mean([run_cnn(SimpleCNN32Filter, cifar_train_images, cifar_train_labels, cifar_test_images, cifar_test_labels, fraction_of_train_samples) for _ in range(2)])
    cnn32_acc_vs_n.append(best_accuracy)
    print("Train Fraction:", str(fraction_of_train_samples))
    print("Accuracy:", str(best_accuracy))
    

# accuracy vs num training samples (two layer cnn (32 filters))
cnn32_two_layer_acc_vs_n = list()
for fraction_of_train_samples in fraction_of_train_samples_space:
    best_accuracy = np.mean([run_cnn(SimpleCNN32Filter2Layers, cifar_train_images, cifar_train_labels, cifar_test_images, cifar_test_labels, fraction_of_train_samples) for _ in range(3)])
    cnn32_two_layer_acc_vs_n.append(best_accuracy)
    print("Train Fraction:", str(fraction_of_train_samples))
    print("Accuracy:", str(best_accuracy))
  '''  
    
plt.rcParams['figure.figsize'] = 13, 10
plt.rcParams['font.size'] = 25
plt.rcParams['legend.fontsize'] = 16.5
plt.rcParams['legend.handlelength'] = 2.5
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

fig, ax = plt.subplots() # create a new figure with a default 111 subplot
#ax.plot(fraction_of_train_samples_space*5000, conv_vs_simplecnn, marker='X', markerfacecolor='red', markersize=10, color='green', linewidth=3, linestyle="--", label="Conv vs SimpleCNN")
ax.plot(fraction_of_train_samples_space*5000, kappa_naive_vs_conv, marker='X', markerfacecolor='red', markersize=10, color='green', linewidth=3, linestyle=":", label="Naive vs Conv")
#ax.plot(fraction_of_train_samples_space*5000, naive_vs_simplecnn, marker='X', markerfacecolor='red', markersize=10, color='green', linewidth=3, label="Naive vs SimpleCNN")

#ax.plot(fraction_of_train_samples_space*5000, conv_rf_predict_proba, marker='X', markerfacecolor='red', markersize=10, color='orange', linewidth=3, linestyle=":", label="Conv RF predict_proba")
#ax.plot(fraction_of_train_samples_space*5000, cnn32_acc_vs_n, marker='X', markerfacecolor='red', markersize=10, color='orange', linewidth=3, linestyle="--", label="CNN (32 filters)")
#ax.plot(fraction_of_train_samples_space*5000, cnn32_two_layer_acc_vs_n, marker='X', markerfacecolor='red', markersize=10, color='orange', linewidth=3, label="CNN Two Layer (32 filters)")

ax.set_xlabel('Number of Train Samples', fontsize=18)
ax.set_xscale('log')
ax.set_xticks([i*5000 for i in fraction_of_train_samples_space])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.set_ylabel('Kappa statistic', fontsize=18)
# ax.set_ylim(0.68, 1)

ax.set_title("3 (cat) vs 9 (truck) Classification Kappas", fontsize=18)
plt.legend()
plt.show()
plt.savefig("cifar_results/" + str(class1) + "_vs_" + str(class2) + " kappa")

