import numpy as np

class LinearRegression:
    def __init__(self):
        self.b_0 = 0  # Intercept
        self.b_1 = 0  # Slope

    def fit(self, X, y):
        X_mean, y_mean = np.mean(X), np.mean(y)
        ssxy = np.sum((X - X_mean) * (y - y_mean))
        ssx = np.sum((X - X_mean) ** 2)
        
        self.b_1 = ssxy / ssx
        self.b_0 = y_mean - self.b_1 * X_mean
        return self.b_0, self.b_1

    def predict(self, X):
        return self.b_0 + self.b_1 * X

if __name__ == '__main__':
    X = np.array([173, 182, 165, 154, 170])
    y = np.array([68, 79, 65, 57, 64])
    
    model = LinearRegression()
    model.fit(X, y)
    print(model.predict(161))
