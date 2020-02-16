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
#Todo link to filters https://www.consumerfinance.gov/data-research/hmda/explore#!/as_of_year=2016,2015,2014&state_code-1=9&loan_purpose=1&section=filters

pd.set_option('display.width', 1000)
pd.options.display.max_rows = 1000000
pd.options.display.max_columns = 1000000

features = [
    'agency_code','applicant_ethnicity','applicant_income_000s','applicant_race_1','applicant_race_2','applicant_race_3',
    'applicant_race_4','applicant_race_5','applicant_sex','application_date_indicator','as_of_year','census_tract_number',
    'co_applicant_ethnicity','co_applicant_race_1','co_applicant_race_2','co_applicant_race_3','co_applicant_race_4',
    'co_applicant_race_5','co_applicant_sex','county_code',
    # 'denial_reason_1','denial_reason_2','denial_reason_3',
    'edit_status',
    'hoepa_status','hud_median_family_income','lien_status','loan_amount_000s','loan_purpose','loan_type','minority_population',
    'msamd','number_of_1_to_4_family_units','number_of_owner_occupied_units','owner_occupancy','population','preapproval',
    'property_type','purchaser_type','rate_spread',
    'issuer',
    'sequence_number','state_code','tract_to_msamd_income']

def load_data():
    data = pd.read_csv('/home/shant/Downloads/hmda/hmda_lar.csv',low_memory=False,header=0)
    data['issuer'] = data[['respondent_id']].apply(lambda col: pd.factorize(col)[0])
    # data = data.drop(data[data.action_taken.isin([4, 5,6])].index)
    data = data[data.action_taken.isin([1,3])]
    data['approved'] = data.action_taken.isin([1, 2, 6, 8]).astype(int)
    data = data.fillna(0)

    X_train,X_test,Y_train,Y_test = train_test_split(data, data.approved, test_size=0.2, random_state=42)
    return X_train,X_test,Y_train,Y_test



class HMDA():
    def __init__(self,num_outputs):
        self.num_outputs = num_outputs
        optimizer = Adam(0.001)
        self.model = self.build_model()
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])
        self.X_train,self.X_test,self.Y_train,self.Y_test = load_data()
        self.max_auc = 0
        self.max_accuracy = 0
        self.losses = []
        self.accuracy = []

    def build_model(self):
        feature_shape = (len(features),)
        feature_in = Input(shape=feature_shape)

        # X = BatchNormalization()(feature_in)
        X = Dense(512,activation=None)(feature_in)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Dense(256,activation=None)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Dense(128,activation=None)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        main_output = Dense(self.num_outputs, activation='softmax')(X)

        model = Model(inputs=feature_in, outputs=main_output)
        model.summary()
        return model

    def train(self,epochs,batch_size,save_interval):
        X_train_approved = self.X_train[self.X_train.approved == 1]
        X_train_denied = self.X_train[self.X_train.approved == 0]
        for epoch in range(epochs):
            half_batch = batch_size//2

            # idx = np.random.randint(0,len(self.X_train),batch_size)
            # x = self.X_train[features].values[idx]
            # y = keras.utils.to_categorical(self.Y_train.values[idx],num_classes=self.num_outputs)
            # loss, accuracy = self.model.train_on_batch(x,y)
            # if epoch % 50 == 0:
            #     print("{} \t [ Loss: {:0.2f} \t Accuracy: {:0.2f}% ]".format(epoch,loss,accuracy*100.0))
            # # print(np.shape(x),np.shape(y))

            idx = np.random.randint(0,len(X_train_approved),half_batch)
            x_a = X_train_approved[features].values[idx]
            y_a = keras.utils.to_categorical(X_train_approved.approved.values[idx],num_classes=self.num_outputs)

            idx = np.random.randint(0,len(X_train_denied),half_batch)
            x_d = X_train_denied[features].values[idx]
            y_d = keras.utils.to_categorical(X_train_denied.approved.values[idx],num_classes=self.num_outputs)

            x = np.asarray(x_a.tolist() + x_d.tolist())
            y = np.asarray(y_a.tolist() + y_d.tolist())
            loss, accuracy = self.model.train_on_batch(x,y)
            if epoch % 50 == 0:
                print("{} \t [ Loss: {:0.2f} \t Accuracy: {:0.2f}% ]".format(epoch,loss,accuracy*100.0))

            # idx = np.random.randint(0,len(X_train_denied),half_batch)
            # x = X_train_denied[features].values[idx]
            # y = y,keras.utils.to_categorical(Y_train_denied.values[idx],num_classes=self.num_outputs)
            # print(np.shape(x),np.shape(y))
            # print(y)


            # x = self.X_train[features].values[idx]
            # y = keras.utils.to_categorical(self.Y_train.values[idx],num_classes=self.num_outputs)

            # self.losses.append(loss)
            # self.accuracy.append(accuracy)
            # if epoch % 50 == 0:
            #     print("{} \t [ Loss: {:0.2f} \t Accuracy: {:0.2f}% ]".format(epoch,loss,accuracy*100.0))

            if epoch % save_interval == 0:
                self.test()
                self.save_model()

    def test(self):
        x = self.X_test[features].values
        y = keras.utils.to_categorical(self.Y_test.values, num_classes=self.num_outputs)
        loss, accuracy = self.model.test_on_batch(x,y)
        color = 'green' if accuracy>=0.90 else 'red'
        print(colored("Test: [ Loss:  {:0.2f} \t Accuracy: {:0.2f}% ]".format(loss,accuracy*100.0),color,attrs=['reverse', 'bold']))

    def save_model(self):
        x = self.X_test
        predictions = self.model.predict(x[features])
        y = keras.utils.to_categorical(self.Y_test.values, num_classes=self.num_outputs)

        x['prediction'] = np.argmax(predictions,1)
        x.to_csv('results.txt',header=True,index=False)
        roc = roc_auc_score(self.Y_test.values, np.argmax(predictions,1))
        print("AUC {:0.2f}".format(roc))

        mislabeled = x[x.approved != x.prediction]
        total = len(x)
        error = len(mislabeled) / total
        accuracy = 1 - error

        if roc > self.max_auc and accuracy > self.max_accuracy*0.95:
            print("Saving model because {:0.2f}/{:0.2f} is greater than {:0.2f}/{:0.2f}".format(roc,accuracy,self.max_auc,self.max_accuracy))
            x[['approved','prediction']].to_csv('results_best.txt', header=True, index=False)
            self.max_auc = roc
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
            self.model.save('model/hmda.h5')


TEST_MODE = False
# TEST_MODE = True
def main():
    hmda = HMDA(2)
    if not os.path.exists('model'):
        os.makedirs('model')

    if os.path.exists('model/hmda.h5'):
        del hmda.model
        hmda.model = load_model('model/hmda.h5')
        print("Loaded model from saved state")

    if TEST_MODE:
        hmda.test()
    else:
        hmda.train(epochs=1000000, batch_size=128, save_interval=500)


if __name__ == '__main__':
    main()