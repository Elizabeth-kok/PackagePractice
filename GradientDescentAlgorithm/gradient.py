# Gradient Descent for Linear Regression
# yhat = wx+b
# loss = (y-yhat)**2 / N
import numpy as np

# Initialise some parameters
# Using numpy to create some x data an y data
# We don't know what w and b are, so we can use gradient descent to find them.
x = np.random.randn(10,1) #return samples from the standard normal distribution
print (x)
print (x.shape)
y = 2*x + np.random.rand()
#Parameters
w = 0.0
b = 0.0
#Hyperparameter
learning_rate=0.01

# Create gradient descent function
# take in x, y, w, b, learning rate
# calculate the partial derivative
# (loss with respect to "w" and loss with respect to "b")
def descent(x, y, w, b, learning_rate):
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]

    # (y-(wx+b))**2 / N <-differentiate this formula
    #loop through x and y at the same time, make an update for dldw and dldb
    for xi, yi in zip(x,y):
        dldw += -2*xi*(yi-(w*xi+b))
        dldb += -2*(yi-(w*xi+b))

    # make updates to w and b
    w = w - learning_rate*(1/N)*dldw
    b = b - learning_rate*(1/N)*dldb

    return w,b


# Iterativaly make updates
# loop through a bunch of epochs/steps (one steps towards the "valley")
for epoch in range(400):
    w,b = descent(x, y, w, b,learning_rate)
    yhat = w*x + b
    loss = np.divide(np.sum((y-yhat) ** 2,axis=0),x.shape[0])
    print (f"{epoch} loss is {loss}, parameters w:{w}, b:{b}")


print(np.sum(np.arange(3),axis=0))
print(np.divide(np.array([5,6,8]),np.arange(3)))
