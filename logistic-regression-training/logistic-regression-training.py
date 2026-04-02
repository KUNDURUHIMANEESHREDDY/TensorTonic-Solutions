import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # 1. Fixed: 'x' to 'X' (Python is case-sensitive)
    m, n = X.shape  
    weights = np.zeros(n)
    bias = 0
    
    for i in range(steps):
        # 2. Forward Pass
        linear_model = np.dot(X, weights) + bias
        # 3. Fixed: renamed 'sigmoid' to '_sigmoid' to match your helper function
        y_predicted = _sigmoid(linear_model)
        
        # 4. Compute Gradients
        dw = (1 / m) * np.dot(X.T, (y_predicted - y))
        db = (1 / m) * np.sum(y_predicted - y)
        
        # 5. Update Parameters
        weights -= lr * dw
        bias -= lr * db
        
        # 6. Optional: Print loss
        if i % 100 == 0:
            # Added a tiny epsilon (1e-15) to prevent log(0) errors
            loss = -np.mean(y * np.log(y_predicted + 1e-15) + (1 - y) * np.log(1 - y_predicted + 1e-15))
            print(f"Step {i}: Loss {loss:.4f}")
            
    return weights, bias