import pickle
import numpy as np
import matplotlib.pyplot as plt

path = './mnist/'
path_test = path + 'mnist_test.csv'
path_train = path + 'mnist_train.csv'


def get_mnist_data_csv():
    test_data = np.loadtxt(path_test, delimiter=",")
    train_data = np.loadtxt(path_train, delimiter=",")

    testi = np.asfarray(test_data[:, 1:])
    testl = np.asfarray(test_data[:, 0])

    traini = np.asfarray(train_data[:, 1:])
    trainl = np.asfarray(train_data[:, 0])
    return traini, trainl, testi, testl


def get_mnist_preproc_csv():
    traini, trainl, testi, testl = get_mnist_data_csv()

    fac = 1.001 * 255
    testi = testi / fac + 0.001
    traini = traini / fac + 0.001

    testi = testi - np.mean(testi, axis=0)
    traini = traini - np.mean(traini, axis=0)

    return traini, trainl, testi, testl


def save_mnist_bin():
    data = get_mnist_preproc_csv()
    with open(path + 'pickled_preproc_mnist.pkl', "bw") as fh:
        pickle.dump(data, fh)
    return


def get_mnist_preproc():
    traini, trainl, testi, testl = None, None, None, None
    with open(path + 'pickled_preproc_mnist.pkl', "br") as fh:
        traini, trainl, testi, testl = pickle.load(fh)

    train_size = len(trainl)
    val_size = int(train_size / 5)
    train_size = train_size - val_size
    vali = traini[train_size:]
    vall = trainl[train_size:]
    traini = traini[:train_size]
    trainl = trainl[:train_size]

    traini = traini.reshape(len(traini), 28, 28, 1)
    vali = vali.reshape(len(vali), 28, 28, 1)
    testi = testi.reshape(len(testi), 28, 28, 1)

    return traini, trainl, vali, vall, testi, testl


def show_mnist_img(index, data):
    img = data[index].reshape((28, 28))
    plt.imshow(img, cmap="Greys")
    plt.show()
    return
