from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle
from sklearn.metrics import precision_recall_curve
import os
from keras.models import model_from_json

main = Tk()
main.title("Sequence classification for credit-card fraud detection")
main.geometry("1300x1200")

global filename
global dataset
global X, Y
global rf_X_train, rf_X_test, rf_y_train, rf_y_test
global lstm_X_train, lstm_X_test, lstm_y_train, lstm_y_test

global rf_precision, rf_recall
global lstm_precision, lstm_recall

def uploadDataset():    
    global filename
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset)+"\n\n")
    sns.heatmap(dataset.corr(), annot = True)
    plt.show()

def preprocess():
    text.delete('1.0', END)
    global X, Y
    global dataset
    global rf_X_train, rf_X_test, rf_y_train, rf_y_test
    global lstm_X_train, lstm_X_test, lstm_y_train, lstm_y_test

    dataset = dataset.sample(frac=1)
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    sm = SMOTE(random_state = 42)
    X, Y = sm.fit_sample(X, Y)
    features = X.shape[1]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X = X[0:50000,0:X.shape[1]]
    Y = Y[0:50000]
    rf_X_train, rf_X_test, rf_y_train, rf_y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

    Y = to_categorical(Y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    lstm_X_train, lstm_X_test, lstm_y_train, lstm_y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    text.insert(END,"Total features extracted from dataset : "+str(features)+"\n")
    text.insert(END,"Total records found in dataset is : "+str(X.shape[0])+"\n")
    text.insert(END,"Total records used to train Random Forest and LSTM : "+str(rf_X_train.shape[0])+"\n")
    text.insert(END,"Total records used to test Random Forest and LSTM : "+str(rf_X_test.shape[0])+"\n")

def runRF():
    global rf_X_train, rf_X_test, rf_y_train, rf_y_test
    global rf_precision, rf_recall
    text.delete('1.0', END)

    rfc = RandomForestClassifier(n_estimators=2,criterion="entropy",max_features="log2",class_weight="balanced",max_leaf_nodes=5)
    rfc.fit(rf_X_train, rf_y_train)
    predict = rfc.predict(rf_X_test)
    random_precision = precision_score(rf_y_test, predict,average='macro') * 100
    random_recall = recall_score(rf_y_test, predict,average='macro') * 100
    rf_precision, rf_recall, thresholds = precision_recall_curve(rf_y_test, predict)
    text.insert(END,"Random Forest Fraud Detection Precision : "+str(random_precision)+"\n")
    text.insert(END,"Random Forest Fraud Detection Recall    : "+str(random_recall)+"\n\n")


def runLSTM():
    global lstm_X_train, lstm_X_test, lstm_y_train, lstm_y_test
    global lstm_precision, lstm_recall
    if os.path.exists('model/lstmmodel.json'):
        with open('model/lstmmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        json_file.close()
        model.load_weights("model/lstmmodel_weights.h5")
        model._make_predict_function()
    else:
        model = Sequential() #creating neural network model object
        #adding LSTM layer with 96 filters to model object which means LSTM will filter dataset values 96 times to select best features and return_sequence=True means
        #this dataset has to filter based on time series
        model.add(LSTM(96, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        #adding drop layer to remove all those features which are irrelevant or having importance or relevant score less than 0.2
        model.add(Dropout(0.2))
        #adding another LSTM layer with 64 filters to reoptimze dataset features and select importnat features which are relevant to predict whether
        #transaction is fraud or not
        model.add(LSTM(64))
        #drop out to remove irrelevant features
        model.add(Dropout(0.2))
        #adding output layer as Y which contains class label as 0 or 1 and it can predict output either 1 or 0
        model.add(Dense(Y.shape[1], activation='softmax'))
        #compile LSTM model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #start trining LSTM for 10 epoch by using X features and Y class label
        hist = model.fit(X, Y, epochs=10, batch_size=16)
        model.save_weights('model/lstmmodel_weights.h5')            
        model_json = model.to_json()
        with open("model/lstmmodel.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/lstmhistory.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()

    predict = model.predict(lstm_X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(lstm_y_test, axis=1)
    lstmPrecision = precision_score(y_test, predict,average='macro') * 100
    lstmRecall = recall_score(y_test, predict,average='macro') * 100
    lstm_precision, lstm_recall, thresholds = precision_recall_curve(y_test, predict)
    text.insert(END,"LSTM Fraud Detection Precision : "+str(lstmPrecision)+"\n")
    text.insert(END,"LSTM Fraud Detection Recall    : "+str(lstmRecall)+"\n\n")


def graph():
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(rf_recall, rf_precision)
    plt.plot(lstm_recall,lstm_precision)
    plt.legend(['RF Base', 'LSTM Base'], loc='upper left')
    plt.title('Precision-Recall curves averaged over all days in the test set')
    plt.show()    

font = ('times', 16, 'bold')
title = Label(main, text='Sequence classification for credit-card fraud detection')
title.config(bg='gold2', fg='thistle1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Credit Card Fraud Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=20,y=150)
processButton.config(font=ff)

rfButton = Button(main, text="Run Random Forest Algorithm", command=runRF)
rfButton.place(x=20,y=200)
rfButton.config(font=ff)

lstmButton = Button(main, text="Run LSTM Algorithm", command=runLSTM)
lstmButton.place(x=20,y=250)
lstmButton.config(font=ff)

graphButton = Button(main, text="Fraud Detection AUC Graph", command=graph)
graphButton.place(x=20,y=300)
graphButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=330,y=100)
text.config(font=font1)

main.config(bg='DarkSlateGray1')
main.mainloop()
