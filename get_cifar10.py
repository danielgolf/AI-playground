import pickle
import numpy as np

path = './cifar10/'


def reshape_image(img):
    new_img = np.zeros((32, 32, 3))
    for d in range(3):
        for i in range(32):
            for j in range(32):
                new_img[i, j, d] = img[1024 * d + 32 * i + j]
    return new_img


def preprocess_cifar10():
    basename = 'data_batch_'
    traini, trainl = [], []
    for i in range(1, 6):
        print(i)
        with open(path + basename + str(i), 'rb') as file:
            batch = pickle.load(file, encoding='bytes')
            images = batch[b'data']
            labels = batch[b'labels']
            for i in range(images.shape[0]):
                traini.append(reshape_image(images[i]))
                trainl.append(labels[i])

    testi, testl = [], []
    with open(path + 'test_batch', 'rb') as file:
        print('test data')
        batch = pickle.load(file, encoding='bytes')
        images = batch[b'data']
        labels = batch[b'labels']
        for i in range(images.shape[0]):
            testi.append(reshape_image(images[i]))
            testl.append(labels[i])

    traini = np.asarray(traini) / 255
    trainl = np.asfarray(trainl)
    testi = np.asarray(testi) / 255
    testl = np.asfarray(testl)

    return traini, trainl, testi, testl


def save_cifar10_bin():
    data = preprocess_cifar10()
    with open(path + 'cifar10_preproc.pkl', "bw") as fh:
        pickle.dump(data, fh)
    return


def load(zero_mean=False):
    traini, trainl, testi, testl = None, None, None, None
    with open(path + 'cifar10_preproc.pkl', "br") as fh:
        traini, trainl, testi, testl = pickle.load(fh)

    if zero_mean:
        traini = traini - np.mean(traini, axis=0)
        testi = testi - np.mean(testi, axis=0)

    train_size = len(trainl)
    val_size = int(train_size / 5)
    train_size = train_size - val_size

    vali = traini[train_size:]
    vall = trainl[train_size:]
    traini = traini[:train_size]
    trainl = trainl[:train_size]

    return traini, trainl, vali, vall, testi, testl


# 0 b'airplane'
# 1 b'automobile'
# 2 b'bird'
# 3 b'cat'
# 4 b'deer'
# 5 b'dog'
# 6 b'frog'
# 7 b'horse'
# 8 b'ship'
# 9 b'truck'