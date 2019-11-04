#  Classify the MNIST digits using a one nearest neighbour classifier and Euclidean distance
import random
import numpy as np
import matplotlib.pyplot as plt
import mnist

def sqDistance(p, q, pSOS, qSOS):
    #  Efficiently compute squared euclidean distances between sets of vectors

    #  Compute the squared Euclidean distances between every d-dimensional point
    #  in p to every d-dimensional point in q. Both p and q are
    #  npoints-by-ndimensions. 
    #  d(i, j) = sum((p(i, :) - q(j, :)).^2)

    d = np.add(pSOS, qSOS.T) - 2*np.dot(p, q.T)
    return d

np.random.seed(1)

#  Set training & testing 
Xtrain, ytrain, Xtest, ytest = mnist.load_data()


def plot_error(n):
    train_size = n
    print("Training size",train_size)
    test_size  = 10000
    Xtrain, ytrain, Xtest, ytest = mnist.load_data()
    Xtrain = Xtrain[0:train_size]
    ytrain = ytrain[0:train_size]

    Xtest = Xtest[0:test_size]
    ytest = ytest[0:test_size]

    #  Precompute sum of squares term for speed
    XtrainSOS = np.sum(Xtrain**2, axis=1, keepdims=True)
    XtestSOS  = np.sum(Xtest**2, axis=1, keepdims=True)

    #  fully solution takes too much memory so we will classify in batches
    #  nbatches must be an even divisor of test_size, increase if you run out of memory 
    if test_size > 1000:
      nbatches = 50
    else:
      nbatches = 5

    batches = np.array_split(np.arange(test_size), nbatches)
    ypred = np.zeros_like(ytest)

    #  Classify
    for i in range(nbatches):
        dst = sqDistance(Xtest[batches[i]], Xtrain, XtestSOS[batches[i]], XtrainSOS)
        closest = np.argmin(dst, axis=1) 
        ypred[batches[i]] = ytrain[closest]

    #  Report
    errorRate = (ypred != ytest).mean()
    print('Error Rate: {:.2f}%\n'.format(100*errorRate))
    error.append(errorRate)
    #image plot
    #plt.imshow(Xtrain[0].reshape(28, 28), cmap='gray')
   # plt.show()
    return error


#plot_error(10000)


# Q1:  Plot a figure where the x-asix is number of training
#      examples (e.g. 100, 1000, 2500, 5000, 7500, 10000), and the y-axis is test error.
size =[ 100, 1000, 2500, 5000, 7500, 10000]
error = []
def train_samples():
  for i in size:
    
    error =plot_error(i)
  return error

error =train_samples()
plt.plot(size,error,marker='o')
plt.title('Training examples / Test error ')
plt.xlabel('Training size')
plt.ylabel('test error')
plt.show()
cv_error = []
# Q2:  plot the n-fold cross validation error for the first 1000 training training examples
Xtrain, ytrain, Xtest, ytest = mnist.load_data()
def cross_validation(n):
  
  train =Xtrain[0:1000]
  label = ytrain[0:1000]
  for i in range(n):
    train1 =np.array_split(train,n)  
    label1 = np.array_split(label,n)
    test = train1.pop(i)
    test_label = label1.pop(i)
    
    c_train = np.concatenate(train1,axis=0)
    c_label = np.concatenate(label1,axis=0)
    C_XtrainSOS = np.sum(c_train**2, axis=1, keepdims=True)
    C_XtestSOS  = np.sum(test**2, axis=1, keepdims=True)
    
    
    test_size = len(test)
    
    if test_size > 1000:
      nbatches = 50
    else:
      nbatches = 5

    batches = np.array_split(np.arange(test_size), nbatches)
    ypred = np.zeros_like(test_label)

    #  Classify
    for i in range(nbatches):
        dst = sqDistance(test[batches[i]], c_train, C_XtestSOS[batches[i]], C_XtrainSOS)
        closest = np.argmin(dst, axis=1) 
        ypred[batches[i]] = c_label[closest]
    #print(ypred)


    #  Report
    errorRate = (ypred != test_label).mean()
    cv_error.append(errorRate)
    #print('Error Rate: {:.2f}%\n'.format(100*errorRate))
    mean_error =  sum(cv_error)/len(cv_error)
  return mean_error
    

cv_list = [3,10,50,100,1000]
cross_validation_error = []
for i in cv_list:
  m=cross_validation(i)
  cross_validation_error.append(m)

plt.plot(cv_list,cross_validation_error,marker='o')
plt.title('n-fold Cross Validation error')
plt.xlabel('Folds')
plt.ylabel('CV error')
