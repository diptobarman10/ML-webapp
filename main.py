import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Building a Machine Learning Classifier with Dipto")

st.write("""
# Explore the Different Classifier
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))

classifer_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return x, y


x,y = get_dataset(dataset_name)
st.sidebar.write("shape of the dataset", x.shape)
st.sidebar.write("Number of classes", len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_sample = st.sidebar.slider("Max Depth", 2, 15)
        n_estimator = st.sidebar.slider("n_estimator", 1, 100)
        params["max_sample"] = max_sample
        params["n_estimator"] = n_estimator
    return params

params = add_parameter_ui(classifer_name)

def get_classifer(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(max_depth=params["max_sample"], n_estimators = params["n_estimator"], random_state= 1253)
    return clf

clf = get_classifer(classifer_name, params)

#classification

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size= 0.2, random_state= 342)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(Y_test, y_pred)
st.write(f"classifer = {classifer_name}")
st.write(f"accuracy = {acc}")


#plot
st.write("""
#Plot graphs
""")

pca = PCA(2)
x_projected =pca.fit_transform(x)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1,x2, c= y , alpha=0.8, cmap ="viridis")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.colorbar()

st.pyplot()