from termcolor import colored
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import keras
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras import backend
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import text_to_word_sequence

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

pd.set_option('display.width', 1000)
pd.options.display.max_rows = 1000000
pd.options.display.max_columns = 1000000
np.set_printoptions(precision=3)


class HMDA():
    def __init__(self,num_outputs):
        self.num_outputs = num_outputs
        optimizer = Adam(0.001)
        self.model = self.build_model()
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])
        self.max_auc = 0
        self.max_accuracy = 0

    def build_model(self):
        feature_shape = (10,)
        feature_in = Input(shape=feature_shape)

        X = Dense(10,activation=None)(feature_in)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        main_output = Dense(self.num_outputs, activation='sigmoid')(X)

        model = Model(inputs=feature_in, outputs=main_output)
        model.summary()
        return model

    def train(self,epochs,batch_size,save_interval):
        for epoch in range(epochs):
            loss, accuracy = self.model.train_on_batch(np.ones((10,10)),np.arange(10))
            weights = self.model.get_weights()[0]
            print(weights[0])
            # exit(0)
            # print(weights)
            if epoch % 50 == 0:
                print("{} \t [ Loss: {:0.2f} \t Accuracy: {:0.2f}% ]".format(epoch,loss,accuracy*100.0))




TEST_MODE = False
# TEST_MODE = True
def main():
    hmda = HMDA(1)
    hmda.train(epochs=1000, batch_size=128, save_interval=500)


if __name__ == '__main__':
    main()