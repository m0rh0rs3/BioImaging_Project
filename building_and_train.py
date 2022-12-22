#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np
import random
from UNet import unet

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1


# inspired by StackOverflow issue: "https://stackoverflow.com/questions/69878085/keras-using-dice-coefficient-loss-function-val-loss-is-not-improving"
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 0.0001) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 0.0001)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def model_building(trainOriginals, trainLabels, testOriginals):
    print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_')
    print('Building the model...')
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    outputs = unet(s)

    model = tf.keras.Model(inputs, outputs)

    adam = tf.keras.optimizers.Adam(learning_rate=0.00006)
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    
    # Modelcheckpoint
    checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')
    ]

    results = model.fit(trainOriginals, trainLabels, validation_split=0.1,
                        batch_size=8, epochs=5, callbacks=callbacks)

    ##########
    idx = random.randint(0, len(trainOriginals))

    preds_train = model.predict(trainOriginals[:int(trainOriginals.shape[0] * 0.9)], verbose=1)
    preds_val = model.predict(trainOriginals[int(trainOriginals.shape[0] * 0.9):], verbose=1)
    preds_test = model.predict(testOriginals, verbose=1)

    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)

    print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_')
    print('Model building DONE ...')

    return preds_train_t

def train():

    print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_')
    print('Preparing Labels...')

    #NB: per trainOriginals e trainLabels, start ed end DEVONO essere uguali
    # flag: se 1 solo converte imaging, se 0 converte segmentation
    # Train set
    # images: metto flag a 1
    trainOriginals = keras_array_of_cases(1, 0 , 150)
    print('trainOriginals done')


    # labels: metto flag a 0
    trainLabels = keras_array_of_cases(0, 0 , 150)
    print('trainLabels done')

    # Test set
    # images: metto flag a 1
    testOriginals = keras_array_of_cases(1,  75, 100)
    print('testOriginals done')


    preds_train_t = model_building(trainOriginals,trainLabels,testOriginals)

    return trainOriginals, trainLabels, testOriginals , preds_train_t

if __name__ == '__main__':
    model_building()
    train()
