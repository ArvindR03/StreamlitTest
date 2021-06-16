from sklearn import neighbors
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd

st.title('Iris Project - Streamlit Demo')
st.sidebar.write("""
# **By Arvind Raghu ✌️**
""")

iris = datasets.load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

st.write("""
## **Dataset**

Here is the dataset:
""")
rowcount = st.sidebar.slider('Number of rows to show:', 1, df.shape[0], value=5)
st.write(df.head(rowcount))

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)
st.write('Shape of the training dataset: ', X_train.shape, ' & ', y_train.shape)
st.write('Shape of the testing dataset: ', X_test.shape, ' & ', y_test.shape)
st.write('The target features are encoded as follows:')
st.write(pd.DataFrame({'target-names':iris.target_names}))

st.write("""
## **Machine Learning - K Nearest Neighbours**

Here we implement the KNN algorithm to classify the data.
""")
n = st.sidebar.slider('Select the number of nearest neighbours:', 1, 10, 5)
knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
correct = []
predicted_feature = []
actual_feature = []

for i in range(0, len(y_test)):
    if y_test[i] == y_pred[i]:
        correct.append('True')
    else:
        correct.append('False')
    predicted_feature.append(iris.target_names[y_pred[i]])
    actual_feature.append(iris.target_names[y_test[i]])

st.write(pd.DataFrame({'predicted': y_pred, 
                        'predicted feature': predicted_feature,
                        'actual': y_test,
                        'actual feature': actual_feature, 
                        'correct': correct}))
x = 0
for i in correct:
    if i == 'True':
        x += 1

accuracy = (x / X_test.shape[0]) * 100
st.write('The accuracy for this model was ', accuracy, '.')

st.write("""
## **Evaluating parameters**

Here we determine which value is best for the `n_neighbours` parameter to the algorithm.
""")
accuracytable = []
    
for n in range(1, 10):
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    correct = []
    predicted_feature = []
    actual_feature = []

    for i in range(0, len(y_test)):
        if y_test[i] == y_pred[i]:
            correct.append('True')
        else:
            correct.append('False')
        predicted_feature.append(iris.target_names[y_pred[i]])
        actual_feature.append(iris.target_names[y_test[i]])

    x = 0
    for i in correct:
        if i == 'True':
            x += 1

    accuracy = (x / X_test.shape[0]) * 100
    accuracytable.append(accuracy)

st.line_chart(accuracytable)