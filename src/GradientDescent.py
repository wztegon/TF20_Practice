import numpy as np
import matplotlib.pyplot as plt

# Size of the points dataset.
m = 200

# Points x-coordinate and dummy value (x0, x1).
# X0 = np.ones((m, 1))
# X1 = np.arange(1, m+1).reshape(m, 1)
# X  = np.hstack((X0, X1))

# Points y-coordinate
# y = np.array([
#     3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
#     11, 13, 13, 16, 17, 18, 17, 19, 21
# ]).reshape(m, 1)

#features normalization
X0 = np.ones((m, 1))
X1_origin = np.arange(1, m+1).reshape(m, 1)
X1_range = max(X1_origin) - min(X1_origin)
X1 = X1_origin / X1_range
X2_origin = X1_origin * X1_origin
X2_range = max(X2_origin) - min(X2_origin)
X2 = X2_origin/X2_range
X  = np.hstack((X0, X1, X2))
y = 2*X1_origin*X1_origin + 3*X1_origin + 5
# The Learning Rate alpha.
alpha = 1

def error_function(theta, X, y):
    '''Error function J definition.'''
    diff = np.dot(X, theta) - y
    return (1./(2*m)) * np.dot(np.transpose(diff), diff)

b1 = True
def gradient_function(theta, X, y):
    '''Gradient of the function J definition.'''
    diff = np.dot(X, theta) - y #(θ0 + θ1*X1) - y
    #return (1./m) * np.dot(np.transpose(X), diff)
    return np.transpose((1. / m) * np.dot(np.transpose(diff), X))
Lost = []
def gradient_descent(X, y, alpha):
    '''Perform gradient descent.'''
    theta = np.ones((3, 1))
    gradient = gradient_function(theta, X, y)
    while not np.all(np.absolute(gradient) <= 1e-7):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
        Lost.append(error_function(theta, X, y)[0, 0])
    return theta

optimal = gradient_descent(X, y, alpha)

# print('optimal:', optimal)
# print('error function:', error_function(optimal, X, y)[0, 0])

print('optimal:', optimal[0],optimal[1]/X1_range, optimal[2]/X2_range)
print('error function:', error_function(optimal, X, y))

# plt.figure(1)
# plt.plot(X1[:,0],y[:,0],"r.")
# f_gradient = optimal[0] + optimal[1] * X1
# x = np.linspace(0,30.,30,endpoint= True)
# plt.plot(X[:,1],f_gradient)
# plt.show()


plt.figure()
plt.plot(X1_origin[:,0],y[:,0] ,"r.")
f_gradient = optimal[0] + optimal[1]/X1_range * X1_origin + optimal[2]/X2_range * X2_origin
# x = np.linspace(0,30.,30,endpoint= True)
plt.plot(X1_origin[:,0],f_gradient)
# plt.show()
plt.figure()
plt.plot(range(len(Lost)),Lost[:],"r.",markersize=1)
plt.show()