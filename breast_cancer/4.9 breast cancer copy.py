from pylab import *
from numpy import *
from scipy import *
# load the data
def load_data():
    # load data
    data = matrix(genfromtxt('breast_cancer_data.csv', delimiter=','))
    X = asarray(data[:,0:8])
    y = asarray(data[:,8])
    y.shape = (size(y),1)
    return (X,y)



def sigmoid(x):
    b = 1/(1+e**(-x))
    return b



def Newtons_method(X,y):
    # use compact notation and initialize
    temp = shape(X)
    temp = ones((temp[0],1))
    X = concatenate((temp,X),1)
    X = X.T
    w = randn(X.shape[0],1)*0.0001
    t = ones((1,y.size))
    w_path =[]
    w_path.append(w)
    # start gradient descent loop
    grad = 1
    k_path = []
    k = 0
    max_its =100
    counter_path =[]
    while linalg.norm(grad) > 10**(-5) and k <= max_its:
        # compute gradient
        z = -sigmoid(-y*dot(X.T,w))
        s = z*y
        h = diag(diag(dot(s,t)))
        f = sum(-z*(1+z))
        grad = dot(dot(X,h),t.T)
        hessian = dot(X,X.T)*f

        # take iteration step
        w = w - dot(inv(hessian),grad)

        # update path containers
        k += 1
        k_path.append(k)
        n = 0
        counter = 0
        while n<699:
            if max(0,-y[n]*dot(X[:,n].T,w))>0:
                counter += 1
            n +=1
        counter_path.append(counter)
        w_path.append(w)

    return (w,counter_path,k_path)

def squard_margin(X,y):
    # use compact notation and initialize
    temp = shape(X)
    temp = ones((temp[0],1))
    X = concatenate((temp,X),1)
    X = X.T
    w = randn(X.shape[0],1)*0.0001
    w_path =[]
    w_path.append(w)
    # start gradient descent loop
    grad = 1
    k_path = []
    k = 0
    max_its =100
    counter1_path =[]
    while linalg.norm(grad) > 10**(-5) and k <= max_its:
        # compute gradient
        m =0
        sum_1 = 0
        while m<699:
            sum_1 = sum_1+(-2*max(0,1-y[m]*dot(X[:,m].T,w))*y[m]*X[:,m])
            m = m+1
        grad = sum_1
        grad.shape =(grad.size,1)
        m = 0
        sum_2 = 0
        while m<699:
            if 1-y[m]*dot(X[:,m].T,w)>0:
                sum_2 =sum_2 +2*dot(X,X.T)
            m = m+1
        hessian = array(sum_2)
        w = w - dot(inv(hessian),grad)
        # update path containers
        k += 1
        k_path.append(k)
        n = 0
        counter = 0
        while n<699:
            if max(0,-y[n]*dot(X[:,n].T,w))>0:
                counter += 1
            n +=1
        counter1_path.append(counter)
        w_path.append(w)

    return (w,counter1_path,k_path)

def plotpic(c,k,c_1,k_1):
    figure(figsize = (7,7))
    xlabel('iterations')
    ylabel('numbers of misclassficaitons')
    plot(k,c,linewidth = 1,color = 'red',label = 'softmax')
    plot(k_1,c_1,linewidth = 1,color = 'blue',label = 'squared margin')
    legend(loc =2)
    show()
    return()



### main loop ###
def main():
    X,y = load_data()
    print y.size
    (w,c,k) = Newtons_method(X,y)
    (w_1,c_1,k_1) = squard_margin(X,y)
    plotpic(c,k,c_1,k_1)
main()

