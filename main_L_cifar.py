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
   Calculate KDE with Gaussian Kernel for CIFAR-100 
   CIFAR-100 datasets are from https://www.cs.toronto.edu/~kriz/cifar.html
"""
import numpy as np
import pickle
import datetime
import KDE_Gauss

starttime = datetime.datetime.now()

u_cifar = pickle.load(open('train','rb'), encoding='latin1')
X_tror = u_cifar['data']
indices = np.random.permutation(np.arange(len(X_tror)))
X_new=[X_tror[i] for i in indices]
X_train = X_new[:10000]
X_valid = X_new[10000:20000]
data_te = pickle.load(open('test','rb'), encoding='latin1')
X_test = data_te['data']

X_train = np.array(X_train)/255
X_valid = np.array(X_valid)/255
X_test = np.array(X_test)/255

#sigma = [0.05, 0.08, 0.1, 0.2, 0.5, 1, 1.5, 2]
sigma = [0.2]
L_cifar = []
for s in sigma:
    l_cifar = KDE_Gauss.L_calcu(X_train,X_valid,s**2)
    L_cifar.append(l_cifar)

print(L_cifar)

endtime = datetime.datetime.now()

print(endtime - starttime)