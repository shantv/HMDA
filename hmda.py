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
from sklearn.metrics import roc_auc_score,roc_curve, auc
# TODO Note: link to data and filter settings
# TODO Note: https://www.consumerfinance.gov/data-research/hmda/explore#!/as_of_year=2016,2015,2014&state_code-1=9&loan_purpose=1&section=filters

pd.set_option('display.width', 1000)
pd.options.display.max_rows = 1000000
pd.options.display.max_columns = 1000000

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

class HMDA():
    def __init__(self,num_outputs):
        self.X_train,self.X_test,self.Y_train,self.Y_test = load_data()
        self.num_features = len(self.X_train.columns)
        self.num_outputs = num_outputs
        optimizer = Adam(0.01)
        self.model = self.build_model()
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])
        self.max_auc = 0
        self.max_accuracy = 0
        self.min_fpr = 1

    def build_model(self):
        feature_shape = (self.num_features,)
        feature_in = Input(shape=feature_shape)

        X = Dense(128,activation=None)(feature_in)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        # X = Dense(256,activation=None)(X)
        # X = BatchNormalization()(X)
        # X = Activation('relu')(X)
        # X = Dense(128,activation=None)(X)
        # X = BatchNormalization()(X)
        # X = Activation('relu')(X)
        main_output = Dense(self.num_outputs, activation='sigmoid')(X)

        model = Model(inputs=feature_in, outputs=main_output)
        model.summary()
        return model

    def train(self,epochs,batch_size,save_interval):
        for epoch in range(epochs):
            # idx = np.random.randint(0,len(self.X_train),batch_size)
            x = self.X_train.values
            y = self.Y_train.values
            history = self.model.fit(x,y,batch_size=batch_size,epochs=1,verbose=False)
            loss = np.mean(history.history['loss'])
            accuracy = np.mean(history.history['acc'])
            print("{} \t [ Loss: {:0.2f} \t Accuracy: {:0.2f}% ]".format(epoch,loss,accuracy*100.0))
            self.test()

    def test(self):
        x = self.X_test
        probs = self.model.predict(x)
        predictions = np.round(probs)

        y = self.Y_test
        y['prediction'] = predictions
        y['probs'] = probs
        y[['approved','prediction','probs']].to_csv('results.txt',header=True,index=False)

        auc_pred = roc_auc_score(self.Y_test.approved.values, predictions)
        auc_prob = roc_auc_score(self.Y_test.approved.values, probs)
        mislabeled = y[y.approved != y.prediction]
        total = len(y)
        error = len(mislabeled) / total
        accuracy = 1 - error
        # print("AUC Pred: {:0.2f} / Prob: {:0.2f}".format(auc_pred,auc_prob))
        fpr_pred, tpr_pred, thresh_pred = roc_curve(y.approved, y.prediction)
        if fpr_pred[1] < self.min_fpr:
            self.min_fpr = fpr_pred[1]

        color = 'green' if accuracy >=0.90 else 'red'
        # print(colored("Test: [ AUC Pred: {:0.2f} / AUC Prob: {:0.2f} \t Accuracy: {:0.2f}% ]".format(auc_pred,auc_prob,accuracy*100.0),color,attrs=['reverse', 'bold']))
        print(colored("Test: [ FPR: {:0.2f} TPR: {:0.2f} \t MinFpr: {:0.2f} \t AUC Pred: {:0.2f} / AUC Prob: {:0.2f} \t Accuracy: {:0.2f}% ]".format(fpr_pred[1],tpr_pred[1],self.min_fpr,auc_pred,auc_prob,accuracy*100.0),color,attrs=['reverse', 'bold']))

        if auc_pred > self.max_auc and accuracy > self.max_accuracy*0.95:
            print("Saving model because {:0.2f}/{:0.2f} is greater than {:0.2f}/{:0.2f}".format(auc_pred,accuracy,self.max_auc,self.max_accuracy))
            y[['approved','prediction','probs']].to_csv('results_best.txt', header=True, index=False)
            self.max_auc = auc_pred
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
            self.model.save('model/hmda.h5')


TEST_MODE = False
# TEST_MODE = True
def main():
    hmda = HMDA(1)
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