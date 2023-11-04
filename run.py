import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class LinearRegression:
    def __init__(self, lr=1e-5, epochs=1000):
        self.bias = 0
        self.losses = []
        self.lr = lr
        self.epochs = epochs

    def predict(self, X):
        return X.dot(self.weight) + self.bias
    

    
    def fit(self, X, y):
        self.n, self.num_of_features = X.shape
        self.weight = np.ones(self.num_of_features)

        for _ in range(self.epochs):
            y_hat = self.predict(X)
            loss = (np.sum((y-y_hat))**2)/self.n
            
            # Calculate partial derivate of weight with respect to loss
            dl_w = (-2)*np.sum((y-y_hat).dot(X)) / self.n
            # Calculate partial derivate of bias with respect to loss
            dl_b = -2*(np.sum(y-y_hat))/self.n

            # Update weight
            self.weight -= self.lr*dl_w
            # Update bias
            self.bias -= self.lr*dl_b

            self.losses.append(loss)

        print(self.weight, self.bias)
        print(self.losses[-1])

    def plot(self):
        plt.scatter(X_data, y_data)
        # Replace with real predicted data
        plt.plot(X_data, (self.weight*X_data + self.bias))
        plt.xlabel('Area')
        plt.ylabel('Price')
        plt.title('House price')
        plt.show()

df = pd.read_csv("Housing.csv")
print(df.head())

#  Single feature data
scaler = MinMaxScaler()

X_data = scaler.fit_transform(np.array(df['area'], dtype=float).reshape(-1,1))
y_data = scaler.fit_transform(np.array(df['price'], dtype=float).reshape(-1,1))


model = LinearRegression()
model.fit(X_data, y_data)
model.plot()