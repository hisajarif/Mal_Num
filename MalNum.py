import tflearn
import os
import numpy as np
import speechData
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

learning_rate = 0.00001
training_iters = 3000 # steps

width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10  # digits

X, Y = speechData.loadDataSet()
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.20, random_state=4)
print("Train data = ",np.asarray(trainX).shape," : ",type(trainX))
print("Train label = ",np.asarray(trainY).shape," : ", type(trainY))
#print(trainY[0:1])
print("Test data = ",np.asarray(testX).shape," : ",type(testX))
print("Test label = ",np.asarray(testY).shape," : ",type(testY))

# Network building
net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')


model = tflearn.DNN(net, tensorboard_verbose=0)
if not os.path.isfile("tflearn.lstm.model.meta"):
    #model.load("tflearn.lstm.model") // for repeated turning by removeing not from if .
    model.fit(trainX, trainY, n_epoch=training_iters, validation_set=(testX, testY), show_metric=True,batch_size=20)
    print("\nSLNO  :  Predict -> Label\n")
    lt = len(testX)
    for i in range(1,lt+1):
        print (i, "\t:  ",np.argmax(model.predict(testX[i-1:i])), " --> ", np.argmax(testY[i-1:i]))
    model.save("tflearn.lstm.model")
else:
    model.load("tflearn.lstm.model")
    print("\n....Model is already trained....\n")
    print("\nSLNO  :  Predict -> Label\n")
    curt = 0
    lt = len(testX)
    for i in range(1,lt+1):
        p = np.argmax(model.predict(testX[i-1:i]))
        v = np.argmax(testY[i-1:i])
        if p == v :
            curt+=1
        print (i, "\t:  ",p, " --> ",v)    
    print("\n\t ACCURACY : ", curt/lt)
    
    #print("\n....prediction on unknown....\n")
    #T = speechData.mfcc_target("predict/")
    #lt = len(T)
    #for i in range(1,lt+1):
    #    print (i, "\t:  ",np.argmax(model.predict(T[i-1:i]))+1)
    


