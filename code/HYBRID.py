import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input, Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle

pick_in = open('pick/title.pickle', 'rb')
df_cop = pickle.load(pick_in)
pick_in.close()
print(df_cop.head())

X_train, X_test, y_train, y_test = train_test_split(df_cop['title'], df_cop['check'], test_size = 0.4, shuffle=True, random_state=43)
print(len(X_train), len(X_test), len(y_train), len(y_test))

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = pad_sequences(X_train, padding='post', maxlen=40)
x_test = pad_sequences(X_test, padding='post', maxlen=40)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

xx_test = x_test[:len(x_test)//4].copy()
yy_test = y_test[:len(y_test)//4].copy()

model = Sequential()
model.add(Input(shape=(40)))
model.add(Embedding(10000, 60))
model.add(Dropout(0.5))
model.add(Conv1D(64, 3, padding='valid', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D())
model.add(Conv1D(64, 3, padding='valid', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D())
model.add(Conv1D(64, 3, padding='valid', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D())
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

history = model_.fit(x_train,y_train,validation_data=(xx_test,yy_test),epochs=15,batch_size=128, verbose=0, callbacks=[tqdm_callback])
model_.evaluate(x_test, y_test, verbose=0)

y_pred=(model.predict(x_test) > 0.5).astype("int32")
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

model_.save('models/Hybrid123.h5')