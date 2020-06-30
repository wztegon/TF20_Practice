import numpy as np
import matplotlib.pyplot as plt

# Size of the points dataset.
m = 40

# Points x-coordinate and dummy value (x0, x1).
X0 = np.ones((m, 1))
X1_origin = np.arange(1, m+1).reshape(m, 1)
X1_range = max(X1_origin) - min(X1_origin)
X1 = X1_origin / X1_range
X2_origin = X1_origin * X1_origin
X2_range = max(X2_origin) - min(X2_origin)
X2 = X2_origin/X2_range
X  = np.hstack((X0, X1, X2))

# Points y-coordinate
# y = np.array([
#     3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
#     11, 13, 13, 16, 17, 18, 17, 19, 21
# ]).reshape(m, 1)
y = 2*X1_origin*X1_origin + 3*X1_origin + 5
# The Learning Rate alpha.
alpha = np.ones((3, 1)) * 1.2

def error_function(theta, X, y):
    '''Error function J definition.'''
    diff = np.dot(X, theta) - y
    return (1./(2*m)) * np.dot(np.transpose(diff), diff)


def gradient_function(theta, X, y):
    '''Gradient of the function J definition.'''
    diff = np.dot(X, theta) - y #(θ0 + θ1*X1) - y
    return (1./batch_size) * np.dot(np.transpose(X), diff)
Lost = []
batch_size = 40
#实现梯度下降
def gradient_descent(X, y, alpha,  batch_size = 10):
    '''Perform gradient descent.'''
    theta = np.array([[0.1], [5], [2]])
    gradient = gradient_function(theta, X, y)
    epoch = 400
    for _ in np.arange(epoch):

    # while not np.all(np.absolute(gradient) <= 1e-5):
        for i in np.arange(m // batch_size):
            for i in range(len(theta)):
                theta[i] = theta[i] - alpha[i] * gradient[i]
                print(gradient)

            gradient = gradient_function(
                theta, X[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size])

            Lost.append(error_function(theta, X, y)[0,0])
    return theta

optimal = gradient_descent(X, y, alpha, batch_size)
print('optimal:', optimal[0],optimal[1]/X1_range, optimal[2]/X2_range)
print('error function:', error_function(optimal, X, y))
plt.figure()
plt.plot(X1_origin[:,0],y[:,0] ,"r.")
f_gradient = optimal[0] + optimal[1]/X1_range * X1_origin + optimal[2]/X2_range * X2_origin
# x = np.linspace(0,30.,30,endpoint= True)
plt.plot(X1_origin[:,0],f_gradient)
# plt.show()
plt.figure()
plt.plot([i for i in range(len(Lost))],Lost[:],"r.",markersize=1)
plt.show()

