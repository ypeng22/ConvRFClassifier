import numpy as np
import matplotlib.pyplot as p


fraction_of_train_samples_space = np.geomspace(0.01, 1, num=8)
p.plot(fraction_of_train_samples_space * 5000, [15, 12, 11, 8, 6, 5, 3, 1])
p.ylabel('number of experiments')
p.xlabel('training samples of each class')
p.title('Conv RF accu > naive RF (out of 45)')