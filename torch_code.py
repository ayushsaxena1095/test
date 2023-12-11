"""
This file and its tests are individualized for NetID asaxen09.
"""
import numpy as np
import torch as tr



def mountain1d(x):

    # Converting input to a PyTorch tensor with requires_grad set to True for automatic differentiation.
    x_tensor = tr.tensor(x, requires_grad=True, dtype=tr.float32)

    # Computing the function value z as given in requirement
    z = 2 * x_tensor**3 - 3 * x_tensor**2 - x_tensor - 2

    # Computing the derivative dz/dx using automatic differentiation.
    z.backward()

    # Extracting the numerical value of the derivative from the tensor and casting it to a float.
    dz_dx = x_tensor.grad.item()

    # Returning the computed float values
    return z.item(), dz_dx
    


def robot(t1, t2):

    # Converting joint angles to PyTorch tensors with requires_grad set to True for automatic differentiation.
    t1_tensor = tr.tensor(t1, requires_grad=True, dtype=tr.float32)
    t2_tensor = tr.tensor(t2, requires_grad=True, dtype=tr.float32)

    # Defining the position of robot in terms of joint angles
    x = 3 * (tr.cos(t1_tensor) * tr.cos(t2_tensor) - tr.sin(t1_tensor) * tr.sin(t2_tensor)) + 3 * tr.cos(t1_tensor)
    y = 3 * (tr.sin(t1_tensor) * tr.cos(t2_tensor) + tr.cos(t1_tensor) * tr.sin(t2_tensor)) + 3 * tr.sin(t1_tensor)

    # Function provided to minimize distance to to a fixed point with x and y values elaborated above
    z = (x - 3)**2 + (y - 0)**2

    # Computin gradients with respect to joint angles
    dz_dt1, dz_dt2 = tr.autograd.grad(z, [t1_tensor, t2_tensor])

    # Converting results back to float
    z = float(z)
    dz_dt1 = float(dz_dt1)
    dz_dt2 = float(dz_dt2)

    # Returning float values
    return z, dz_dt1, dz_dt2
    
   
    
def neural_network(W1, W2, W3):
    
    # Converting weight matrices to PyTorch tensors with requires_grad set to True for automatic differentiation.
    W1_tensor = tr.tensor(W1, requires_grad=True, dtype=tr.float32)
    W2_tensor = tr.tensor(W2, requires_grad=True, dtype=tr.float32)
    W3_tensor = tr.tensor(W3, requires_grad=True, dtype=tr.float32)

    # Defining input vectors as PyTorch tensors.
    x1 = tr.tensor([1.0, -1.0], dtype=tr.float32)
    x2 = tr.tensor([-1.0, 1.0, -1.0], dtype=tr.float32)
    x3 = tr.tensor([-1.0, 1.0, 1.0, 1.0], dtype=tr.float32)

    # Forward pass through the network.
    z3 = tr.matmul(W3_tensor, tr.tanh(x3))
    z2 = tr.matmul(W2_tensor, tr.tanh(x2 + z3))
    y = tr.matmul(W1_tensor, tr.tanh(x1 + z2))

    # Calculating the squared error.
    e = (y - 2.0) ** 2

    # Backward pass to compute gradients.
    e.backward()

    # Extracting the gradients of the error with respect to each weight matrix.
    de_dW1 = W1_tensor.grad.numpy()
    de_dW2 = W2_tensor.grad.numpy()
    de_dW3 = W3_tensor.grad.numpy()

    # Detaching and converting results back to numpy arrays with float32 type.
    y = y.detach().numpy().astype(np.float32)
    e = e.detach().numpy().astype(np.float32)

    # Returning float values
    return y, e, de_dW1, de_dW2, de_dW3


    

if __name__ == "__main__":

    # start with small random weights
    W1 = np.random.randn(1,2).astype(np.float32) * 0.01
    W2 = np.random.randn(2,3).astype(np.float32) * 0.01
    W3 = np.random.randn(3,4).astype(np.float32) * 0.01
    
    # do several iterations of gradient descent
    for step in range(100):
        
        # evaluate loss and gradients
        y, e, dW1, dW2, dW3 = neural_network(W1, W2, W3)
        if step % 10 == 0: print("%d: error = %f" % (step, e))

        # take step
        eta = .1/(step + 1)
        W1 -= dW1 * eta
        W2 -= dW2 * eta
        W3 -= dW3 * eta

