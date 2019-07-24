import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.layers import Dense, Activation

N = 250 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes


class KerasMinNN:
  def __init__(self, hidden_neurons=100, reg=3e-4):
    # hyperparameter
    self.hidden_neurons = hidden_neurons
    self.reg = reg
    # setup model
    self.model = keras.Sequential()
    # hidden layer, input 2-dim
    self.model.add(
        Dense(
            hidden_neurons,
            input_shape=(D,),
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(reg),
            kernel_initializer=keras.initializers.RandomNormal(),
            bias_initializer='zeros'
        )
    )
    # output layer 3 neurons / classes, softmax probs
    self.model.add(
        Dense(
            K,
            activation='softmax',
            kernel_regularizer=keras.regularizers.l2(reg),
            kernel_initializer=keras.initializers.RandomNormal(),
            bias_initializer='zeros'
        )
    )
    # training model: use adam, cross-entropy loss
    self.model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

  def train(self, X, y, iterations=400):
    y_cat = keras.utils.to_categorical(y, num_classes=None)
    self.history = self.model.fit(
        X, y_cat,
        epochs=iterations
    )
    return

  def plot_acc_loss(self):
    # summarize history for accuracy
    plt.plot(self.history.history['acc'])
    # plt.plot(self.history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    plt.plot(self.history.history['loss'])
    # plt.plot(self.history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return

  def get_accurancy(self, X, y):
    scores = self.model.predict(X)
    predicted_class = np.argmax(scores, axis=1)
    return np.mean(predicted_class == y)

  def visualize_model(self):
    # get weights and biases
    params = self.model.get_weights()
    W, b, W2, b2 = params[0], params[1], params[2], params[3]
    # plot the resulting classifier
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    Z = np.dot(np.maximum(
        0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
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
# end KerasMinNN


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
  # plot_data(X, y)

  kmnn = KerasMinNN()
  old_acc = kmnn.get_accurancy(X, y)
  kmnn.train(X, y)
  trained_acc = kmnn.get_accurancy(X, y)

  print("untrained accuracy: %.2f\ntrained accuracy: %.2f" % (old_acc, trained_acc))
  # kmnn.visualize_model()
  kmnn.plot_acc_loss()
