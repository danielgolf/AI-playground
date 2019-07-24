import keras
import numpy as np
import get_cifar10 as gcf
import keras.layers as layers
# import matplotlib.pyplot as plt

### --- hyperparameterrs --- ###

epochs = 16
batch_size = 64
num_classes = 10

reg = 3e-3

### --- hyperparams end --- ###

### --- setup data --- ###

traini, trainl, vali, vall, testi, testl = gcf.load(zero_mean=True)

trainl = keras.utils.to_categorical(trainl, num_classes=None)
vall = keras.utils.to_categorical(vall, num_classes=None)
testl = keras.utils.to_categorical(testl, num_classes=None)

### --- end setup --- ###

### --- define model --- ###

model = keras.Sequential()

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

# Score trained model.
scores = model.evaluate(testi, testl, verbose=1)
print('\nTest loss untrained:', scores_untrained[0])
print('Test accuracy untrained:', scores_untrained[1])
print('\nTest loss:', scores[0])
print('Test accuracy:', scores[1])

### --- end training --- ###

### --- save model --- ###

model.summary()

json_string = model.to_json()
with open('./cifar10/cifar10_model.json', 'w') as file:
  file.write(json_string + '\n')
model.save_weights('./cifar10/cifar10_weights.hdf5')

### --- end save --- ###

### --- show learning --- ###

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

### --- end show --- ###
