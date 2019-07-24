import numpy as np
import keras
import keras.layers as layers
from get_mnist import get_mnist_preproc

### --- hyperparameterrs --- ###

epochs = 48
batch_size = 64
num_classes = 10

reg = 3e-3

### --- hyperparams end --- ###

### --- setup data --- ###

traini, trainl, vali, vall, testi, testl = get_mnist_preproc()

trainl = keras.utils.to_categorical(trainl, num_classes=None)
vall = keras.utils.to_categorical(vall, num_classes=None)
testl = keras.utils.to_categorical(testl, num_classes=None)

### --- end setup --- ###

### --- define model --- ###

model = keras.Sequential()

# TODO: regularzation
model.add(
    layers.Conv2D(
        input_shape=traini.shape[1:],
        activation='relu',
        filters=8,
        kernel_size=3,
        padding='same',
        kernel_regularizer=keras.regularizers.l2(reg)
    )
)

model.add(
    layers.Conv2D(
        activation='relu',
        filters=8,
        kernel_size=3,
        padding='same',
        kernel_regularizer=keras.regularizers.l2(reg)
    )
)

model.add(
    layers.MaxPooling2D(
        pool_size=2
    )
)

model.add(
    layers.Conv2D(
        activation='relu',
        filters=8,
        kernel_size=3,
        padding='same',
        kernel_regularizer=keras.regularizers.l2(reg)
    )
)

model.add(
    layers.Conv2D(
        activation='relu',
        filters=8,
        kernel_size=3,
        padding='same',
        kernel_regularizer=keras.regularizers.l2(reg)
    )
)

model.add(
    layers.Flatten()
)

model.add(
    layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=keras.regularizers.l2(reg)
    )
)

### --- end definition --- ###

### --- training --- ###

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Score untrained model.
scores_untrained = model.evaluate(testi, testl, verbose=1)

history = model.fit(
    traini, trainl,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(vali, vall),
    shuffle=True
)

print('Test loss untrained:', scores_untrained[0])
print('Test accuracy untrained:', scores_untrained[1])
# Score trained model.
scores = model.evaluate(testi, testl, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

### --- end training --- ###

### --- save model --- ###

model.summary()

json_string = model.to_json()
with open('./mnist/mnist_model.json', 'w') as file:
  file.write(json_string + '\n')
model.save_weights('./mnist/mnist_weights.hdf5')

### --- end save --- ###
