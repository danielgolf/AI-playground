import numpy as np
import matplotlib.pyplot as plt

N = 250 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes


class OneHiddenNN:
  def __init__(self, hidden_neurons=100, reg=1e-3, step=1e-0):
    # initialize parameters randomly
    self.W = 0.01 * np.random.randn(D, hidden_neurons)
    self.W2 = 0.01 * np.random.randn(hidden_neurons, K)
    self.b = np.zeros((1, hidden_neurons))
    self.b2 = np.zeros((1, K))
    # hyperparameter
    self.hidden_neurons = hidden_neurons
    self.reg = reg
    self.step = step

  def get_scores(self, X):
    hidden_layer = np.maximum(0, np.dot(X, self.W) + self.b)
    return np.dot(hidden_layer, self.W2) + self.b2

  def get_probs(self, X):
    scores = self.get_scores(X)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

  def get_reg_loss(self):
    reg_loss = 0.5 * self.reg * np.sum(self.W * self.W)
    reg_loss += 0.5 * self.reg * np.sum(self.W2 * self.W2)
    return reg_loss

  def get_data_loss(self, X, y):
    num_examples = X.shape[0]
    probs = self.get_probs(X)
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs) / num_examples
    return data_loss

  def get_loss(self, X, y):
    return self.get_reg_loss() + self.get_data_loss(X, y)

  def get_gradient(self, X, y):
    num_examples = X.shape[0]
    hidden_layer = np.maximum(0, np.dot(X, self.W) + self.b)

    dscores = self.get_probs(X)
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples               # TODO: Needed?

    dW2 = np.dot(hidden_layer.T, dscores)
    dW2 += self.reg * self.W2
    db2 = np.sum(dscores, axis=0, keepdims=True)

    dhidden = np.dot(dscores, self.W2.T)
    dhidden[hidden_layer <= 0] = 0

    dW = np.dot(X.T, dhidden)
    dW += self.reg * self.W
    db = np.sum(dhidden, axis=0, keepdims=True)

    return dW, db, dW2, db2

  def train(self, X, y, iterations=15000):
    for i in range(iterations):
      dW, db, dW2, db2 = self.get_gradient(X, y)
      self.W += -self.step * dW
      self.b += -self.step * db
      self.W2 += -self.step * dW2
      self.b2 += -self.step * db2
      if i % 1000 == 0:
        print("Iteration %d: loss %f" % (i, self.get_loss(X, y)))
    return

  def print_accurancy(self, X, y):
    predicted_class = np.argmax(self.get_scores(X), axis=1)
    print("training accuracy: %.2f" % (np.mean(predicted_class == y)))
    return

  def visualize_model(self):
    # plot the resulting classifier
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    Z = np.dot(np.maximum(
        0, np.dot(np.c_[xx.ravel(), yy.ravel()], self.W) + self.b), self.W2) + self.b2
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    # fig = plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=5)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
    #fig.savefig('spiral_linear.png')
    return
# end OneHiddenNN


def generate_data():
  X = np.zeros((N*K, D))  # data matrix (each row = single example)
  y = np.zeros(N*K, dtype='uint8')  # class labels

  for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2  # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

  return X, y


def plot_data(X, y):
  # lets visualize the data:
  plt.scatter(X[:, 0], X[:, 1], c=y, s=5)
  plt.show()
  return


if __name__ == "__main__":
  X, y = generate_data()
  plot_data(X, y)
  mnn = OneHiddenNN()
  mnn.print_accurancy(X, y)
  mnn.train(X, y)
  mnn.print_accurancy(X, y)
  mnn.visualize_model()
