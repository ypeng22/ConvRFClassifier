import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import torch.nn
import torch

class ConvRFClassifier(object):
    
    #TODO: pass in custom ConvRF args
    def __init__(self, layers = 1, kernel_size = (5,), stride = (2,), n_estimators = 100, num_outputs = 10):
        if not (len(kernel_size) == layers and len(stride) == layers):
            raise Exception('Length of kernel_sizes and strides must be same as layers')
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel_forests = [None] * layers
        self.num_outputs = num_outputs
        self.n_estimators = n_estimators
        self.layers = layers
        self.RF = RandomForestClassifier(n_estimators = 100, n_jobs = -1)


    def _convolve(self, images, kernel_size, stride, labels=None, flatten=False):
        if images.shape[0] != images.shape[1]:
            raise Exception('Only square images are allowed')
        batch_size, in_dim, _, num_channels = images.shape
        out_dim = int((in_dim - kernel_size) / stride) + 1  # calculate output dimensions

        # create matrix to hold the chopped images
        out_images = np.zeros((batch_size, out_dim, out_dim,
                               kernel_size, kernel_size, num_channels))

        curr_y = out_y = 0
        # move kernel vertically across the image
        while curr_y + kernel_size <= in_dim:
            curr_x = out_x = 0
            # move kernel horizontally across the image
            while curr_x + kernel_size <= in_dim:
                # chop images
                out_images[:, out_x, out_y] = images[:, curr_x:curr_x +
                                                     kernel_size, curr_y:curr_y + kernel_size, :]
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1

        if flatten:
            out_images = out_images.reshape(batch_size, out_dim, out_dim, -1)

        out_labels = None
        if labels is not None:
            out_labels = np.zeros((batch_size, out_dim, out_dim))
            out_labels[:, ] = labels.reshape(-1, 1, 1)

        return out_images, out_labels, out_dim


    def convolve_fit(self, images, labels):
        #convolve self.layers times
        for layer in range(self.layers):
            sub_images, sub_labels, out_dim = self._convolve(images, self.kernel_size[layer], self.stride[layer], \
                                                             labels=labels, flatten=True)
            
            #initiate kernel_forest
            self.kernel_forests[layer] = [[0]*out_dim for _ in range(out_dim)]
            convolved_image = np.zeros((images.shape[0], out_dim, out_dim, self.num_outputs))
            
            for i in range(out_dim):
                for j in range(out_dim):
                    self.kernel_forests[layer][i][j] = RandomForestClassifier(n_estimators=self.num_outputs, max_depth=6, n_jobs = -1)
                    self.kernel_forests[layer][i][j].fit(sub_images[:, i, j], sub_labels[:, i, j])
                    convolved_image[:, i, j] = self.kernel_forests[layer][i][j].apply(sub_images[:, i, j])
            images = convolved_image
        return images


    def convolve_predict(self, images):
        
        if not self.kernel_forests[0]:
            raise Exception("Should fit training data before predicting")
        
        for layer in range(self.layers):
            sub_images, _, out_dim = self._convolve(images, self.kernel_size[layer], self.stride[layer], flatten=True)
    
            kernel_predictions = np.zeros((images.shape[0], out_dim, out_dim, self.num_outputs))
            for i in range(out_dim):
                for j in range(out_dim):
                    kernel_predictions[:, i, j] = self.kernel_forests[layer][i][j].apply(sub_images[:, i, j])
            images = kernel_predictions
            
        return images  
 
    
    def fit(self, images, labels):
        im = self.convolve_fit(images, labels)
        im = im.reshape(len(images), -1)
        self.RF.fit(im, labels)

       
    def predict(self, images):
        im = self.convolve_predict(images)
        im = im.reshape(len(images), -1)
        return self.RF.predict(im)
    
    
    def predict_proba(self, images):
        im = self.convolve_predict(self, images)
        return self.RF.predict_proba(im)