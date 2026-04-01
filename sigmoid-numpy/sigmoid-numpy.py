def sigmoid(x):
    x = np.array(x)  # Fix: convert list to numpy array
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )