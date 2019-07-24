import keras
import numpy as np

from get_mnist import show_mnist_img
from get_mnist import get_mnist_preproc

json_string = None
with open('./mnist/mnist_model.json', 'r') as file:
    json_string = file.readline()
model = keras.models.model_from_json(json_string)
model.load_weights('./mnist/mnist_weights.hdf5')

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

traini, trainl, vali, vall, testi, testl = get_mnist_preproc()
testl = keras.utils.to_categorical(testl, num_classes=None)
# TODO: mit unprocessed csv daten testen

# Score trained model.
# scores = model.evaluate(testi, testl, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

while True:
    index = np.random.randint(0, testl.shape[0])
    show_mnist_img(index, testi)
    prediction = model.predict(testi[index].reshape(1, 28, 28, 1))
    print('prediction:', np.where(prediction[0].round(1) == 1.0))
