#Load Iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import math 
import numpy as np
import array
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

#load data into iris variable
iris = load_iris()

#this is splitting step
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)


#this function includes general transaction. 
def predict_(k_neighbor, x_train, x_test, y_train):
    
    result = np.array([])
    for data in x_test:

        
        arr = np.array(euclidean(x_train, y_train, data))
        value = accordingto_knn_select_one_class(k_neighbor, arr[:k_neighbor])
        
        result = np.append(result, value)

    return result #returns an list that is result of classification




#this function is euclidean algorithms. it returns distance of each samples
def euclidean(x_train, y_train, x_test):
    
    
    similarity = []
    
    for i, i_y in zip(x_train, y_train):
        index = 0
        sum_ = 0 
        for j in i : 
            
            subtract = x_test[index] - j 
            pow_ = math.pow(subtract, 2)
            
            sum_ += pow_
            
            index+=1
            
        similarity.append((math.sqrt(sum_),i_y))
    
    similarity.sort(key=lambda x:x[0])
    return similarity
            
#this function do things that calculate neighbor according to how much constraint k-neighbor,
#and it decide class of species and returns name of class
def accordingto_knn_select_one_class(k_neighbor, predict_target_value):
    
    classes = [(0.0,0), (1.0,0), (2.0,0)]
    classes = np.array(classes)
    
    for i in range(len(predict_target_value)):
        (a,b) = predict_target_value[i]
#         print(a,b)
        if b == 0.0:
            classes[0][1] +=1
        elif b == 1.0:
            classes[1][1] +=1
        elif b == 2.0:
            classes[2][1] +=1
    
    out = sorted(classes, key=lambda x:x[1], reverse=True)[0]
#     print(out[0])
    return out[0]



y_pred_1 = predict_(1, X_train, X_test, y_train) #this variable represents that all predicted samples
y_pred_2 = predict_(3, X_train, X_test, y_train) 
y_pred_3 = predict_(5, X_train, X_test, y_train)

y_true = np.array(y_test) #this variable represents y_test

#below operations calculate accuracy score.
print('According to 1-neighbor, accuracy score : ' , accuracy_score(y_true, y_pred_1))
print('According to 3-neighbor, accuracy score : ' , accuracy_score(y_true, y_pred_2))
print('According to 5-neighbor, accuracy score : ' , accuracy_score(y_true, y_pred_3))
 

from sklearn.neighbors import KNeighborsClassifier

def calculate_accuracy(k_neighbor):    
    classifier = KNeighborsClassifier(n_neighbors=k_neighbor)
    classifier.fit(X_train, y_train)
    y_pred2 = classifier.predict(X_test)
    return accuracy_score(y_test, y_pred2)




#below operations calculate accuracy score.
print('According to 1-neighbor, accuracy score : ' , calculate_accuracy(1))
print('According to 3-neighbor, accuracy score : ' , calculate_accuracy(3))
print('According to 5-neighbor, accuracy score : ' , calculate_accuracy(5))
 