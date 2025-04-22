import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self):
        self.params = None  # Simplified initialization

    def fit(self, X, y):
        X_bias = np.c_[np.ones(len(X)), X]  # Combined bias addition
        self.params = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y  # Combined calculation
        return self.params
    
    def predict(self, X):
        X_test = np.c_[np.ones(len(X)), X]
        sigmoid = 1 / (1 + np.exp(-X_test @ self.params))
        y_hat = (sigmoid >= 0.5).astype(int)  # Simplified thresholding
        return sigmoid, y_hat

if __name__ == '__main__':
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    sig, y_pred = model.predict(X_test)
    
    print(f'Predictions: {y_pred}\nProbabilities: {sig}')
    print(f'Test value: {y_test[14]}, Predicted: {y_pred[14]}, Probability: {sig[14]}')
