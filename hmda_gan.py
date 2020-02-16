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
from keras.layers.advanced_activations import LeakyReLU

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# TODO Note: link to data and filter settings
# TODO Note: https://www.consumerfinance.gov/data-research/hmda/explore#!/as_of_year=2016,2015,2014&state_code-1=9&loan_purpose=1&section=filters

pd.set_option('display.width', 1000)
pd.options.display.max_rows = 1000000
pd.options.display.max_columns = 1000000
MODEL_NAME = 'hmda_gan'

categorical_features = [
    'agency_code', 'applicant_ethnicity', 'respondent_id',
    'applicant_race_1', 'applicant_race_2', 'applicant_race_3', 'applicant_race_4', 'applicant_race_5',
    'applicant_sex',
    'co_applicant_ethnicity',
    'co_applicant_race_1', 'co_applicant_race_2', 'co_applicant_race_3', 'co_applicant_race_4',
    'co_applicant_race_5',
    'co_applicant_sex',
    'county_code',
    'hoepa_status',
    'lien_status',
    'loan_type',
    'owner_occupancy',
    'property_type',
    'preapproval',
    'purchaser_type',
    'state_code',
    'loan_purpose'
]
discrete_features = [
    'applicant_income_000s',
    'application_date_indicator',
    'as_of_year',
    'census_tract_number',
    'edit_status',
    'hud_median_family_income',
    'loan_amount_000s',
    'minority_population',
    'msamd',
    'number_of_1_to_4_family_units',
    'number_of_owner_occupied_units',
    'population',
    'rate_spread',
    'sequence_number',
    'tract_to_msamd_income'
    ]

def load_data():
    data = pd.read_csv('/home/shant/Downloads/hmda/hmda_lar.csv',low_memory=False,header=0)
    # data['issuer'] = data[['respondent_id']].apply(lambda col: pd.factorize(col)[0])
    data = data[data.action_taken.isin([1,3])]
    data['approved'] = data.action_taken.isin([1]).astype(int)
    data = data.fillna(0)
    categorical = pd.get_dummies(data[categorical_features],columns=categorical_features)
    x = pd.concat([data[discrete_features],categorical],axis=1)
    y = data[['approved']]

    X_train,X_test,Y_train,Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return X_train,X_test,Y_train,Y_test

class HMDA_GAN():
    def __init__(self,num_outputs,model_name):
        self.model_name = model_name
        self.num_outputs = num_outputs
        self.X_train,self.X_test,self.Y_train,self.Y_test = load_data()
        self.num_features = len(self.X_train.columns)
        self.max_auc = 0
        self.max_accuracy = 0

        optimizer = Adam(0.0002,0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generates loan application features
        z = Input(shape=(100,))
        gen_x = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated features as input and determines validity
        valid,predictions = self.discriminator(gen_x)

        # The combined model (stacked generator and discriminator) takes
        # noise as input => generates features => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        noise_shape = (100,)
        noise_in = Input(shape=noise_shape)

        X = Dense(256,activation=None)(noise_in)
        X = LeakyReLU(alpha=0.2)(X)
        X = BatchNormalization(momentum=0.8)(X)
        X = Dense(512,activation=None)(X)
        X = LeakyReLU(alpha=0.2)(X)
        X = BatchNormalization(momentum=0.8)(X)
        X = Dense(1024,activation=None)(X)
        X = LeakyReLU(alpha=0.2)(X)
        X = BatchNormalization(momentum=0.8)(X)

        gen_output = Dense(self.num_features, activation='tanh',name='gen_output')(X)

        model = Model(inputs=noise_in, outputs=gen_output)
        model.summary()
        return model

    def build_discriminator(self):
        feature_shape = (self.num_features,)
        feature_in = Input(shape=feature_shape)

        X = Dense(512,activation=None)(feature_in)
        X = LeakyReLU(alpha=0.2)(X)
        X = Dense(256,activation=None)(X)
        X = LeakyReLU(alpha=0.2)(X)

        valid_output = Dense(1, activation='sigmoid',name='valid')(X)
        label_output = Dense(self.num_outputs,activation='sigmoid',name='prediction')(X)
        model = Model(inputs=feature_in, outputs=[valid_output,label_output])
        model.summary()
        return model


    def train(self,epochs,batch_size,save_interval):
        half_batch = batch_size // 2
        # Rescale values
        x_train = self.X_train
        x_norm = ((x_train - x_train.mean()) / (x_train.max() - x_train.min())).fillna(0)
        x_test = self.X_test
        self.x_test_norm = ((x_test - x_test.mean()) / (x_test.max() - x_test.min())).fillna(0)

        for epoch in range(epochs):
            idx = np.random.randint(0,len(self.X_train),half_batch)
            x = x_norm.values[idx]
            y = self.Y_train.values[idx]

            noise = np.random.uniform(-1, 1, (half_batch, 100))

            # Generate a half batch of loan application features
            gen_x = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real,d_valid_loss_real,d_pred_loss_real,d_acc_real,d_pred_acc = self.discriminator.train_on_batch(x, [np.ones((half_batch,1)),y])
            d_loss_fake,_,_,d_acc_fake,_ = self.discriminator.train_on_batch(gen_x, [np.zeros((half_batch, 1)),y])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            d_acc = 0.5 * np.add(d_acc_real, d_acc_fake)

            # Train the generator with the objective of tricking the discriminators valid output target (valid_y = 1.0)
            noise = np.random.uniform(-1, 1, (batch_size, 100))
            valid_y = np.array([1] * batch_size)
            g_loss = self.combined.train_on_batch(noise, valid_y)

            if epoch % 50 == 0:
                print("%d [D loss: %f, valid acc.: %.2f%%, pred acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss, 100 * d_acc, 100 * d_pred_acc, g_loss))

            if epoch % save_interval == 0:
                self.test()

    def test(self, SAVE = True):
        x = self.x_test_norm
        valid,probs = self.discriminator.predict(x)
        predictions = np.round(probs)
        # print(predictions)
        results = self.Y_test[['approved']]
        results['prediction'] = predictions
        results['probs'] = probs
        results[['approved','prediction','probs']].to_csv('results_gan.txt',header=True,index=False)
        auc_pred = roc_auc_score(self.Y_test.values, predictions)
        auc_prob = roc_auc_score(self.Y_test.values, probs)

        mislabeled = results[results.approved != results.prediction]
        total = len(results)
        error = len(mislabeled) / total
        accuracy = 1 - error

        color = 'green' if accuracy >=0.90 else 'red'
        print(colored("Test: [ AUC Pred: {:0.2f} / AUC Prob: {:0.2f} \t Accuracy: {:0.2f}% ]".format(auc_pred,auc_prob,accuracy*100.0),color,attrs=['reverse', 'bold']))

        if auc_pred > self.max_auc and accuracy > self.max_accuracy*0.95 and SAVE == True:
            print("Saving model because {:0.2f}/{:0.2f} is greater than {:0.2f}/{:0.2f}".format(auc_pred,accuracy,self.max_auc,self.max_accuracy))
            results[['approved','prediction','probs']].to_csv('results_gan_best.txt', header=True, index=False)
            self.max_auc = auc_pred
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
            # self.model.save('model/'+self.model_name+'.h5')
            self.discriminator.save('model_gan/d.h5')
            self.generator.save('model_gan/g.h5')
            self.combined.save('model_gan/c.h5')


TEST_MODE = False
# TEST_MODE = True
def main():
    gan = HMDA_GAN(1,MODEL_NAME)
    if not os.path.exists('model_gan'):
        os.makedirs('model_gan')

    if os.path.exists('model_gan/c.h5'):
        del gan.generator
        del gan.discriminator
        gan.generator = load_model('model_gan/g.h5')
        gan.discriminator = load_model('model_gan/d.h5')
        print("Loaded model from saved state")

    if TEST_MODE:
        gan.test(SAVE=False)
    else:
        gan.train(epochs=1000000, batch_size=32, save_interval=500)


if __name__ == '__main__':
    main()