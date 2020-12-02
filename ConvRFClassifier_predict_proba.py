import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

#problem with pickling
class ConvRFClassifier(BaseEstimator):
    
   
    def __init__(self, layers = 1, kernel_size = (5,), stride = (2,), n_estimators = 100, num_outputs = 10):
        """A convolutional random forest classifier.
        Initializes a ConvRFClassifier.
        Parameters
        ----------
        layers : int, default = 1
            How many layers of convolutions.
        kernel_size: tuple of integers, default = (5,)
            Size of each kernel at each layer. Length should be equal to layers.
        stride: tuple of integers, default = (5,)
            Number of pixels skipped per image segment at each layer. 
            Length should be equal to layers.
        n_estimators: integer, default = 100
            Number of trees in final random forest classifier.
        num_outputs: integer, default = 10
            Number of trees in the forest that convolve image segments.
        Returns
        -------
        """
        
        if not (len(kernel_size) == layers and len(stride) == layers):
            raise Exception('Length of kernel_sizes and strides must be same as layers')
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_outputs = num_outputs
        self.n_estimators = n_estimators
        self.layers = layers
        #initialize Random Forest array to hold kernels
        self.kernel_forests = [None] * self.layers


    def segment(self, images, kernel_size, stride, labels=None, flatten=True):
        """Segment the images into multiple chunks.
        Parameters
        ----------
        images : array-like, of shape (n_samples, dimension_1, dimension_2, channel)
            Input data. dimension_1 should be equal to dimension_2.
        kernel_size: integer
            Size of each image segment
        stride: integer
            Number of pixels skipped per image segment.
        labels: array-like, of shape (n_samples, dimension_1, dimension_2, 1), default=None
            True class labels for each image segment.
        flatten: boolean, default=True
            Whether image segments should be flattened. Default is true.
        Returns
        -------
        out_images : array-like, of shape (n_samples, dimension_1, dimension_2, dimension_3)
            Image segments. dimension_1 = dimension_2 and dimension_3 = kernel_size * 
            kernel_size * channels if flatten is true.
        out_labels : array-like, of shape (n_samples, dimension_1, dimension_2)
            True image labels for each image segment. Same for all segments taken from
            the same image.
        out_dim : integer
            Dimension of output equal to dimension_1 and dimension_2. Calculated by
            (dimension of input - kernel_size) / stride + 1.
        """
        
        if images.shape[1] != images.shape[2]:
            raise Exception('Only square images are allowed')
            
        #get dimensions of input and output
        batch_size, in_dim, _, num_channels = images.shape
        out_dim = int((in_dim - kernel_size) / stride) + 1  # calculate output dimensions

        # create matrix to hold the chopped imagesO
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
                #increment x and x index
                curr_x += stride
                out_x += 1
            #increment y and y index
            curr_y += stride
            out_y += 1

        #flatten image into vector by row major order
        if flatten:
            out_images = out_images.reshape(batch_size, out_dim, out_dim, -1)
            
        #flatten labels if any
        out_labels = None
        if labels is not None:
            out_labels = np.zeros((batch_size, out_dim, out_dim))
            out_labels[:, ] = labels.reshape(-1, 1, 1)

        return out_images, out_labels, out_dim


    def convolve_fit(self, images, labels):
        """Fit the random forests for convolution.
        Parameters
        ----------
        images : array-like, of shape (n_samples, dimension_1, dimension_2, dimension_3)
            Input data. dimension_1 should be equal to dimension_2. dimension_3 should
            be flattened vector of image segments.
        labels: array-like, of shape (n_samples, dimension_1, dimension_2)
            True class labels for each image segment.
        Returns
        -------
        images : array-like, of shape (n_samples, dimension_1, dimension_2, num_outputs)
            convoluted results for each image segment for each tree.
        """

        #initialize Random Forest Classifier that does the final classification
        self.random_forest = RandomForestClassifier(n_estimators = self.n_estimators, n_jobs = -1)
        #convolve self.layers times
        for layer in range(self.layers):
            #get input image segments
            sub_images, sub_labels, out_dim = self.segment(images, self.kernel_size[layer], self.stride[layer], \
                                                             labels=labels, flatten=True)
            
            #initiate another array to hold kernels for each image segment
            self.kernel_forests[layer] = [[0]*out_dim for _ in range(out_dim)]
            convolved_image = np.zeros((images.shape[0], out_dim, out_dim, 1))
            
            #iterate through length and width of convolved image segments
            for i in range(out_dim):
                for j in range(out_dim):
                    #initialize Random Forests to act like kernels
                    self.kernel_forests[layer][i][j] = RandomForestClassifier(n_estimators=self.num_outputs, max_depth=6, n_jobs = -1)
                    #fit the kernels
                    self.kernel_forests[layer][i][j].fit(sub_images[:, i, j], sub_labels[:, i, j])
                    #convolution step, image segment -> int
                    convolved_image[:, i, j] = self.kernel_forests[layer][i][j].predict_proba(sub_images[:, i, j])[:,1][:,None]
            images = convolved_image
        return images


    def convolve_predict(self, images):
        """Convolve images using random forests.
        Parameters
        ----------
        images : array-like, of shape (n_samples, dimension_1, dimension_2, dimension_3)
            Input data. dimension_1 should be equal to dimension_2. dimension_3 should
            be flattened vector of image segments.
        Returns
        -------
        images : array-like, of shape (n_samples, dimension_1, dimension_2, num_outputs)
            convoluted results for each image segment for each tree.
        """
        #check if forest is fit
        if not self.kernel_forests[0]:
            raise Exception("Should fit training data before predicting")
        
        #repeat for each layer
        for layer in range(self.layers):
            #segment image
            sub_images, _, out_dim = self.segment(images, self.kernel_size[layer], self.stride[layer], flatten=True)
            
            #initialize convolved images
            kernel_predictions = np.zeros((images.shape[0], out_dim, out_dim, 1))
            for i in range(out_dim):
                for j in range(out_dim):
                    kernel_predictions[:, i, j] = self.kernel_forests[layer][i][j].predict_proba(sub_images[:, i, j])[:,1][:,None]
                    #apply convolution
                    #kernel_predictions[:, i, j] = self.kernel_forests[layer][i][j].apply(sub_images[:, i, j])
            images = kernel_predictions           
        return images  
 
    
    def fit(self, images, labels):
        """Fit estimator.
        Parameters
        ----------
        X : array-like, of shape (n_samples, dimension_1, dimension_2, channels)
            Input data.  Array of 3 dimensional images.
        y : array-like, 1D numpy array
            Labels
        Returns
        -------
        self : object
        """
        #fit the kernels and convolve images
        im = self.convolve_fit(images, labels)
        im = im.reshape(len(images), -1)
        #fit the classifier
        self.random_forest.fit(im, labels)

       
    def predict(self, images):
        """Predict class for X.
        Parameters
        ----------
        X : array-like, of shape (n_samples, dimension_1, dimension_2, channels)
            Input data.  Array of 3 dimensional images.
        Returns
        -------
        y : array-like of shape (n_samples,)
            Returns predicted class labels for samples X.
        """
        #convolve the images
        im = self.convolve_predict(images)
        im = im.reshape(len(images), -1)
        #predict the labels
        return self.random_forest.predict(im)
    
    
    def predict_proba(self, images):
        """Predict class probabilities for X.
        The predicted class probabilities of an input sample are computed as
        the mean predicted class of the trees in the forest.
        Parameters
        ----------
        X : array_like of shape (n_samples, dimension_1, dimension_2, channels)
            Input data.  Array of 3 dimensional images.
        Returns
        -------
        p : array-like of shape (n_samples,)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        #convolve the images
        im = self.convolve_predict(self, images)
        im = im.reshape(len(images), -1)
        #predict the probability of labels
        return self.random_forest.predict_proba(im)