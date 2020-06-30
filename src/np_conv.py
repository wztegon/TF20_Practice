import numpy as np


N_SAMPLES = 100
TEST_SIZE = 0.2


layer_list =  np.array([5, 4, 4, 3, 2, 1])


weight_lst = []#save each layer weight
bias_lst = []#save each layer bias
activate_fun = ["relu", "relu", "relu", "relu", "relu", "sigmoid"]

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

def creat_weight_bias(lst, single_sameple):
	single_d = single_sameple.size
	for i, element in enumerate(lst):
		weight = np.random.normal(0, 1, single_d * element).reshape(single_d, element)
		weight_lst.append(weight)
		bias = np.random.normal(0, 1, element)
		bias_lst.append(bias)
		single_d = element
	
def forward(lst, single_sameple):
	input = single_sameple
	for i, element in enumerate(lst):
		out = np.matmul(weight_lst[i].T, input) + bias_lst[i]
		if activate_fun[i] == "relu":
			out = relu(out)
		elif activate_fun[i] == "sigmoid":
		    out = sigmoid(out)
		else:
			raise Exception('Non-supported activation function')
		input = out
		
	return out
def all_forward(lst, all_sameple):
	all_out = []
	for single_sameple in all_sameple:
		out = forward(lst, single_sameple)
		all_out.append(out)
	return all_out


def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
	# number of examples
	m = A_prev.shape[1]
	
	# selection of activation function
	if activation is "relu":
		backward_activation_func = relu_backward
	elif activation is "sigmoid":
		backward_activation_func = sigmoid_backward
	else:
		raise Exception('Non-supported activation function')
	
	# calculation of the activation function derivative
	dZ_curr = backward_activation_func(dA_curr, Z_curr)
	
	# derivative of the matrix W
	dW_curr = np.dot(dZ_curr, A_prev.T) / m
	# derivative of the vector b
	db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
	# derivative of the matrix A_prev
	dA_prev = np.dot(W_curr.T, dZ_curr)
	
	return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
	grads_values = {}
	
	# number of examples
	m = Y.shape[1]
	# a hack ensuring the same shape of the prediction vector and labels vector
	Y = Y.reshape(Y_hat.shape)
	
	# initiation of gradient descent algorithm
	dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
	
	for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
		# we number network layers from 1
		layer_idx_curr = layer_idx_prev + 1
		# extraction of the activation function for the current layer
		activ_function_curr = layer["activation"]
		
		dA_curr = dA_prev
		
		A_prev = memory["A" + str(layer_idx_prev)]
		Z_curr = memory["Z" + str(layer_idx_curr)]
		
		W_curr = params_values["W" + str(layer_idx_curr)]
		b_curr = params_values["b" + str(layer_idx_curr)]
		
		dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
			dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
		
		grads_values["dW" + str(layer_idx_curr)] = dW_curr
		grads_values["db" + str(layer_idx_curr)] = db_curr
	
	return grads_values

def update(params_values, grads_values, nn_architecture, learning_rate):

    # iteration over network layers
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values


def get_cost_value(Y_hat, Y):
    # number of examples
    m = Y_hat.shape[1]
    # calculation of the cost according to the formula
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()

def train(X, Y, nn_architecture, epochs, learning_rate, verbose=False, callback=None):
	# initiation of neural net parameters
	#params_values = init_layers(nn_architecture, 2)
	params_values = creat_weight_bias(layer_list, X_train[0])
	# initiation of lists storing the history
	# of metrics calculated during the learning process
	cost_history = []
	accuracy_history = []
	
	# performing calculations for subsequent iterations
	for i in range(epochs):
		# step forward
		Y_hat, cashe = full_backward_propagation(X, params_values, nn_architecture)
		
		# calculating metrics and saving them in history
		cost = get_cost_value(Y_hat, Y)
		cost_history.append(cost)
		accuracy = get_accuracy_value(Y_hat, Y)
		accuracy_history.append(accuracy)
		
		# step backward - calculating gradient
		grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
		# updating model state
		params_values = update(params_values, grads_values, nn_architecture, learning_rate)
		
		if (i % 50 == 0):
			if (verbose):
				print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
			if (callback is not None):
				callback(i, params_values)
	
	return params_values
def main():
	creat_weight_bias(layer_list, X_train[0])
	all_out = all_forward(layer_list, X_train)
	print(len(all_out))
if __name__ == '__main__':
    main()