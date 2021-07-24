import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Input
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import preprocessing, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from kerastuner.tuners import RandomSearch
import pickle

pick_in = open('pick/title.pickle', 'rb')
df_cop = pickle.load(pick_in)
pick_in.close()
print(df_cop.head())

X_train, X_test, y_train, y_test = train_test_split(df_cop['title'], df_cop['check'], test_size = 0.4, shuffle=True)
print(len(X_train), len(X_test), len(y_train), len(y_test))
X_train = pad_sequences(X_train, padding='post', maxlen=40)
X_test = pad_sequences(X_test, padding='post', maxlen=40)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

XX_test = X_test[:len(X_test)//5].copy()
yy_test = y_test[:len(y_test)//5].copy()

def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(40)))
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation=hp.Choice('activation_fun', ['relu', 'tanh']),
                               kernel_initializer=hp.Choice('distribution', ['glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model
    
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=3,
    directory='project',
    project_name='ANN_test_1')

history = tuner.search(X_train, y_train,
                       epochs=10,
                       batch_size=128,
                       validation_data=(XX_test, yy_test))
print(tuner.results_summary())
best_model_dict = tuner.get_best_hyperparameters()[0].values
print(best_model_dict)
best_model = tuner.get_best_models()[0]
best_model.evaluate(X_test, y_test)
best_model.summary()

pick = open('pick/ANN.pickle', 'wb')
pickle.dump(tuner,pick)
pick.close()
best_model.save('models/ANN.h5')