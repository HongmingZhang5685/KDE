#Python3
#=======================================================================
# Copyright (C) 2018 Hongming Zhang   Email: zhanghm5685@163.com
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#=======================================================================
"""
   Visualize MNIST      
   MNIST datasets are from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

u_mnist = pickle._Unpickler(open('mnist.pkl','rb'))
u_mnist.encoding = 'latin1'
data_mnist = u_mnist.load()
X_tror= data_mnist[0][0]
indices = np.random.permutation(np.arange(len(X_tror)))
X_new=[X_tror[i] for i in indices]
X_train = X_new[:10000]
X_valid = X_new[10000:20000]
X_test, y_test = data_mnist[2]
X_train = np.array(X_train)
X_valid = np.array(X_valid)

def display_data(X):
    figs, axes = plt.subplots(20, 20, figsize=(6, 6))
    for i in range(20):
        for j in range(20):
            axes[i, j].imshow(-X[i + 20 * j].reshape(28, 28), cmap='gray', interpolation='none')
            axes[i, j].axis('off')
    plt.show()

display_data(X_train)    