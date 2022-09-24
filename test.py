
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import os
from keras.layers import LSTM
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import os
from keras.models import model_from_json

dataset = pd.read_csv("Dataset/creditcard.csv")
dataset = dataset.sample(frac=1)



dataset = dataset.values

X = dataset[:,0:dataset.shape[1]-1]
Y = dataset[:,dataset.shape[1]-1]
sm = SMOTE(random_state = 42)
X, Y = sm.fit_sample(X, Y)


print(X.shape)
print(Y.shape)
print(X)
co,un = np.unique(Y,return_counts=True)
print(co)
print(un)

#X = normalize(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

X = X[0:50000,0:X.shape[1]]
Y = Y[0:50000]

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y, test_size = 0.2, random_state = 0)

rfc = RandomForestClassifier(n_estimators=2,criterion="entropy",max_features="log2",class_weight="balanced",max_leaf_nodes=5)
rfc.fit(X_train1, y_train1)
predict = rfc.predict(X_test1)
random_precision = precision_score(y_test1, predict,average='macro') * 100
random_recall = recall_score(y_test1, predict,average='macro') * 100

precision, recall, thresholds = precision_recall_curve(y_test1, predict)
#plt.plot(recall, precision)
#plt.show()

print(random_precision)
print(random_recall)

Y = to_categorical(Y)
X = X.reshape((X.shape[0], X.shape[1], 1))
print(X.shape)
print(Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
print(y_train)
print(y_test)

'''

model = Sequential()
model.add(LSTM(96, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X, Y, epochs=10, batch_size=16)
model.save_weights('model/lstmmodel_weights.h5')            
model_json = model.to_json()
with open("model/lstmmodel.json", "w") as json_file:
    json_file.write(model_json)
json_file.close()    
f = open('model/lstmhistory.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()
'''


if os.path.exists('model/lstmmodel.json'):
    with open('model/lstmmodel.json', "r") as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
    json_file.close()
    model.load_weights("model/lstmmodel_weights.h5")
    model._make_predict_function()


predict = model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test = np.argmax(y_test, axis=1)
random_precision = precision_score(y_test, predict,average='macro') * 100
random_recall = recall_score(y_test, predict,average='macro') * 100

print(random_precision)
print(random_recall)

precision1, recall1, thresholds = precision_recall_curve(y_test, predict)

plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recall, precision)
plt.plot(recall1,precision1)
plt.legend(['RF Base', 'LSTM Base'], loc='upper left')
plt.title('Covid Classification Deep Learning Accuracy & Loss Graph')
plt.show()    





