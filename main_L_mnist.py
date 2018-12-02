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
   Calculate KDE with Gaussian Kernel for MNIST
   MNIST datasets are from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
"""
import numpy as np
import pickle
import datetime
import KDE_Gauss

starttime = datetime.datetime.now()

u_mnist = pickle._Unpickler(open('mnist.pkl','rb'))
u_mnist.encoding = 'latin1'
data_mnist = u_mnist.load()
X_tror = data_mnist[0][0]
indices = np.random.permutation(np.arange(len(X_tror)))
X_new=[X_tror[i] for i in indices]
X_train = X_new[:10000]
X_valid = X_new[10000:20000]
X_test = data_mnist[2][0]
X_train = np.array(X_train)
X_valid = np.array(X_valid)
X_test = np.array(X_test)   

sigma = [0.05, 0.08, 0.1, 0.2, 0.5, 1, 1.5, 2]

L_mnist = []
for s in sigma:
    l_mnist = KDE_Gauss.L_calcu(X_train,X_valid,s**2)
    L_mnist.append(l_mnist)


print(L_mnist)

endtime = datetime.datetime.now()

print(endtime - starttime)