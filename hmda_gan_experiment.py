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
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import text_to_word_sequence
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import CustomObjectScope

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve, auc
# TODO Note: link to data and filter settings
# TODO Note: https://www.consumerfinance.gov/data-research/hmda/explore#!/as_of_year=2016,2015,2014&state_code-1=9&loan_purpose=1&section=filters

pd.set_option('display.width', 1000)
pd.options.display.max_rows = 1000000
pd.options.display.max_columns = 1000000
MODEL_NAME = 'hmda_gan_exp'

categorical_features = [
    'agency_code',
    'applicant_ethnicity',
    'respondent_id',
    'applicant_race_1',
    'applicant_race_2',
    'applicant_race_3',
    'applicant_race_4',
    'applicant_race_5',
    'applicant_sex',
    'co_applicant_ethnicity',
    'co_applicant_race_1',
    'co_applicant_race_2',
    'co_applicant_race_3',
    'co_applicant_race_4',
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
    # categorical = pd.get_dummies(data[categorical_features])
    categorical = pd.get_dummies(data[categorical_features],columns=categorical_features)
    x = pd.concat([data[discrete_features],categorical],axis=1)
    y = data[['approved']]
    # print(y.approved.unique())
    # print(np.shape(x),np.shape(y))
    # exit(0)

    # x = ((x - x.mean()) / (x.max() - x.min())).fillna(0)

    X_train,X_test,Y_train,Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return X_train,X_test,Y_train,Y_test

def disc_loss(x,y):
    # print("X1",x)
    # print("Y0",y)
    loss = backend.tf.reduce_mean(backend.tf.nn.sigmoid_cross_entropy_with_logits(logits=x,labels=y))
    # print("Loss shape",tf.shape(loss))
    return loss

class HMDA_GAN():
    def __init__(self,sess,num_outputs,model_name):
        self.sess = sess
        self.model_name = model_name
        self.num_outputs = num_outputs
        self.X_train,self.X_test,self.Y_train,self.Y_test = load_data()
        self.num_features = len(self.X_train.columns)

        self.X_train_norm = ((self.X_train - self.X_train.mean()) / (self.X_train.max() - self.X_train.min())).fillna(0)
        self.X_test_norm = ((self.X_test - self.X_test.mean()) / (self.X_test.max() - self.X_test.min())).fillna(0)

        self.max_auc = 0
        self.max_accuracy = 0
        self.min_fpr = 1
        optimizer = Adam(0.0001,0.5)

        self.discriminator,self.feature_in,self.valid_in = self.build_discriminator()
        self.discriminator.compile(loss=disc_loss,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        weights = self.discriminator.trainable_weights
        self.d_label = self.discriminator.output[0]
        self.d_valid = self.discriminator.output[1]
        self.d_loss = disc_loss(self.d_label,self.valid_in)
        self.disc_grads = tf.gradients(self.d_loss,weights)
        grads = zip(self.disc_grads, weights)
        self.optimize_d = tf.train.AdamOptimizer(0.0001).apply_gradients(grads)

        self.generator,self.noise_in,self.gen_label_in = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.gen_x = self.generator([self.noise_in,self.gen_label_in])

        self.discriminator.trainable = False
        label,valid = self.discriminator([self.gen_x,self.gen_label_in])

        self.combined = Model(inputs=[self.noise_in,self.gen_label_in], outputs=valid,name='combined')
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        weights = self.combined.trainable_weights
        self.g_loss = disc_loss(self.combined.output,self.gen_label_in)
        self.gen_grads = tf.gradients(self.g_loss,weights)
        grads = zip(self.gen_grads, weights)
        self.optimize_g = tf.train.AdamOptimizer(0.0001).apply_gradients(grads)

        self.sess.run(tf.global_variables_initializer())

        # print("Com",self.combined.loss_functions)
        # print("G",self.generator.loss_functions)
        # print("D",self.discriminator.loss_functions)

    def build_generator(self):
        noise_shape = (100,)
        noise_in = Input(shape=noise_shape,name='gen_noise_in')
        label_shape = (1,)
        label_in = Input(shape=label_shape,name='gen_label_in')
        x_in = keras.layers.concatenate([noise_in, label_in])

        X = Dense(32,activation=None)(x_in)
        X = LeakyReLU(alpha=0.2)(X)
        X = BatchNormalization(momentum=0.8)(X)
        X = Dense(64,activation=None)(X)
        X = LeakyReLU(alpha=0.2)(X)
        X = BatchNormalization(momentum=0.8)(X)
        X = Dense(128,activation=None)(X)
        X = LeakyReLU(alpha=0.2)(X)
        X = BatchNormalization(momentum=0.8)(X)

        gen_output = Dense(self.num_features, activation='tanh',name='gen_output')(X)
        # gen_label = Dense(1,activation='sigmoid',name='gen_label')(X)
        # model = Model(inputs=[noise_in,label_in], outputs=gen_output)
        model = Model(inputs=[noise_in,label_in], outputs=gen_output,name='generator')
        model.summary()
        return model,noise_in,label_in

    def build_discriminator(self):
        feature_shape = (self.num_features,)
        valid_shape = (1,)
        feature_in = Input(shape=feature_shape,name="disc_feature_in")
        valid_in = Input(shape=valid_shape,name="disc_valid_in")
        x_in = keras.layers.concatenate([feature_in, valid_in])

        X = Dense(128,activation=None)(x_in)
        X = LeakyReLU(alpha=0.2)(X)
        X = Dense(64,activation=None)(X)
        X = LeakyReLU(alpha=0.2)(X)

        label_output = Dense(self.num_outputs,activation='sigmoid',name='prediction')(X)
        valid_output = Dense(1, activation='sigmoid',name='valid')(label_output)

        model = Model(inputs=[feature_in,valid_in], outputs=[label_output,valid_output],name='discriminator_features')
        # model_label = Model(inputs=[feature_in,label_in], outputs=label_output,name='discriminator_label')
        model.summary()
        # model_label.summary()
        return model,feature_in,valid_in

        # model_feature = Model(inputs=[feature_in,label_in], outputs=valid_output,name='discriminator_features')
        # model_label = Model(inputs=[feature_in,label_in], outputs=label_output,name='discriminator_labels')
        # model_feature.summary()
        # model_label.summary()
        # return model_feature,model_label


    def train(self,epochs,batch_size,save_interval):
        half_batch = batch_size // 2

        for epoch in range(epochs):
            idx = np.random.randint(0,len(self.X_train_norm),half_batch)
            x = self.X_train_norm.values[idx]
            y = self.Y_train.values[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            d_loss_real,_ = self.sess.run([self.d_loss,self.optimize_d], feed_dict={
                self.feature_in: x,
                self.valid_in: np.ones_like(y)
            })
            gen_x = self.sess.run(self.gen_x,feed_dict={
                self.noise_in:noise,
                self.gen_label_in: y
            })
            d_loss_fake,_ = self.sess.run([self.d_loss,self.optimize_d], feed_dict={
                self.feature_in: gen_x,
                self.valid_in: np.zeros_like(y),
            })
            d_loss = 0.5*(d_loss_fake+d_loss_real)

            noise = np.random.normal(0, 1, (half_batch, 100))
            g_loss,_ = self.sess.run([self.g_loss,self.optimize_g], feed_dict={
                self.noise_in: noise,
                self.gen_label_in: np.ones_like(y)
            })

            if epoch % 50 == 0:
                print("%d [D loss: %f, ] [G loss: %f,]" % (epoch, d_loss, g_loss))
                # print("%d [D loss: %f, valid acc.: %.2f%%, pred acc.: %.2f%%] [G loss: %f, valid: %f, pred: %f]" % (epoch, d_loss, 100 * d_acc, 100 * d_pred_acc, g_loss,g_loss_valid,g_loss_label))

            if epoch % save_interval == 0:
                self.test()

    def test(self, SAVE = True):
        x = self.X_test_norm
        y = self.Y_test[['approved']]
        probs = self.sess.run(self.d_label,feed_dict={
            self.feature_in: x,
            self.valid_in: np.ones_like(y)
        })
        print("probs",probs)
        # label,probs = self.discriminator.predict([x,y])

        predictions = np.round(probs)
        results = self.Y_test
        results['prediction'] = predictions
        results['probs'] = probs
        results[['approved','prediction','probs']].to_csv('results_gan_exp_v2.txt',header=True,index=False)
        auc_pred = roc_auc_score(self.Y_test.approved.values, predictions)
        auc_prob = roc_auc_score(self.Y_test.approved.values, probs)
        fpr_pred, tpr_pred, thresh_pred = roc_curve(results.approved, results.prediction)

        mislabeled = results[results.approved != results.prediction]
        total = len(results)
        error = len(mislabeled) / total
        accuracy = 1 - error

        color = 'green' if accuracy >=0.90 else 'red'
        print(colored("Test: [ FPR: {:0.2f} TPR: {:0.2f} \t MinFpr: {:0.2f} \t AUC Pred: {:0.2f} / AUC Prob: {:0.2f} \t Accuracy: {:0.2f}% ]".format(fpr_pred[1],tpr_pred[1],self.min_fpr,auc_pred,auc_prob,accuracy*100.0),color,attrs=['reverse', 'bold']))
        if fpr_pred[1] < self.min_fpr:
            self.min_fpr = fpr_pred[1]

        if auc_pred > self.max_auc and accuracy > self.max_accuracy*0.95 and SAVE == True:
            print("Saving model because {:0.2f}/{:0.2f} is greater than {:0.2f}/{:0.2f}".format(auc_pred,accuracy,self.max_auc,self.max_accuracy))
            results[['approved','prediction','probs']].to_csv('results_gan_exp_best_v2.txt', header=True, index=False)
            self.max_auc = auc_pred
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
            self.discriminator.save('model_gan_exp/d.h5')
            self.generator.save('model_gan_exp/g.h5')
            self.combined.save('model_gan_exp/c.h5')


TEST_MODE = False
# TEST_MODE = True
def main():
    sess = tf.Session()
    backend.set_session(sess)
    gan = HMDA_GAN(sess,1,MODEL_NAME)
    if not os.path.exists('model_gan_exp'):
        os.makedirs('model_gan_exp')

    if os.path.exists('model_gan_exp/c.h5'):
        del gan.generator
        del gan.discriminator
        with CustomObjectScope({'disc_loss':disc_loss}):
            gan.generator = load_model('model_gan_exp/g.h5')
            gan.discriminator = load_model('model_gan_exp/d.h5')
            print("Loaded model from saved state")

    if TEST_MODE:
        gan.test(SAVE=False)
    else:
        gan.train(epochs=1000000, batch_size=128, save_interval=500)


if __name__ == '__main__':
    main()