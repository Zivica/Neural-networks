import numpy as np


class NeuralNetwork:
    # To prevent gradient vanishing we use sqrt(1/...)
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(1 / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(1 / hidden_size)
        self.b2 = np.zeros((output_size, 1))

    # Good for binary classifications, causes gradient vanishing in the deep networks
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def compute_loss(self, Y):
        m = Y.shape[1]
        A2_clipped = np.clip(self.A2, 1e-15, 1 - 1e-15)
        return -(1 / m) * np.sum(Y * np.log(A2_clipped) + (1 - Y) * np.log(1 - A2_clipped))

    def backward(self, X, Y):
        m = Y.shape[1]
        dZ2 = self.A2 - Y
        self.dW2 = (1 / m) * np.dot(dZ2, self.A1.T)
        self.db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(self.W2.T, dZ2) * self.A1 * (1 - self.A1)
        self.dW1 = (1 / m) * np.dot(dZ1, X.T)
        self.db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    def update_params(self, learning_rate):
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1

    def train(self, X, Y, learning_rate, epochs):
        for epoch in range(epochs):
            A2 = self.forward(X)
            loss = self.compute_loss(Y)
            self.backward(X, Y)
            self.update_params(learning_rate)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.4f}")


# Dataset (XOR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Y = np.array([[0, 1, 1, 0]])

# Initialize and train
nn = NeuralNetwork(input_size=2, hidden_size=8, output_size=1)
nn.train(X, Y, learning_rate=1.0, epochs=20000)

# Test
print("\nPredictions:", np.round(nn.forward(X)))