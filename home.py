#import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import precision_score, recall_score, auc,roc_curve, f1_score
import warnings
warnings.filterwarnings('ignore')

st.title("Comparitive Analysis of ML Classifers")

tab1, tab2 = st.tabs(["Dataset wise :clipboard:", "Classifier wise :chart:"])

with tab1:
    st.header("Dataset wise")
    
    dataset_name = st.selectbox("Select Dataset",("Iris", "Breast Cancer", "Wine"))
    classifier_name = st.selectbox("Select Classifier",("KNN", "SVM", "Random Forest"))
    
    with st.sidebar:
        st.write(f"**About {dataset_name} Dataset:**")
        if dataset_name == "Iris":
            st.write(" Iris Dataset contains four features (length and width of sepals and petals) of 50 samples of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). These measures were used to create a linear discriminant model to classify the species.")
        elif dataset_name == "Breast Cancer":
            st.write(" Breast dataset is a comprehensive dataset that contains nearly all the PLCO study data available for breast cancer incidence and mortality analyses. For many women the trial documents multiple breast cancers, however, this file only has data on the earliest breast cancer diagnosed in the trial.")
        elif dataset_name == "Wine":
            st.write(" Wine dataset are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars.")
        
        st.write("")

        st.write(f"**About {classifier_name} Classifier:**")
        if classifier_name == "KNN":
            st.write(" k-nearest neighbors algorithm is a non-parametric supervised learning method first developed by Evelyn Fix and Joseph Hodges in 1951, and later expanded by Thomas Cover. It is used for classification and regression. In both cases, the input consists of the k closest training examples in a data set.")
        elif classifier_name == "SVM":
            st.write("Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees")
        elif classifier_name == "Random Forest":
            st.write("Support vector machines are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis")

    def get_dataset(dataset_name):
        if dataset_name == "Iris":
            data = datasets.load_iris()
        elif dataset_name == "Breast Cancer":
            data = datasets.load_breast_cancer()
        elif dataset_name == "Wine":
            data = datasets.load_wine()
        X = data.data
        y = data.target
        return X,y
    
    X, y = get_dataset(dataset_name)
    st.write("Shape of dataset", X.shape)
    st.write("Number of classes", len(np.unique(y)))

    def add_parameter_ui(clf_name):
        params = dict()
        if clf_name == "KNN":
            K = st.slider("K", 1, 15)
            params["K"] = K
        elif clf_name == "SVM":
            C = st.slider("C", 0.01,10.0)
            params["C"] = C
        elif clf_name == "Random Forest":
            max_depth = st.slider("max_depth", 2, 15)
            n_estimators = st.slider("n_estimators", 1, 100)
            params["max_depth"]=max_depth
            params["n_estimators"]=n_estimators
        return params

    params = add_parameter_ui(classifier_name)

    def get_classifier(clf_name, params):
        if clf_name == "KNN":
            clf = KNeighborsClassifier(n_neighbors=params["K"])
        elif clf_name == "SVM":
            clf = SVC(C=params["C"])
        elif clf_name == "Random Forest":
            clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=1234)
        return clf

    clf = get_classifier(classifier_name, params)

    
    # Classification 
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1234)

    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    st.subheader(f"**Classifier = {classifier_name}**")
    st.subheader(f"Accuracy = {acc}")

    
# Viz (Principal Component Analysis - Feature reduction)
    pca=PCA(2)
    X_projected = pca.fit_transform(X)

    x1 = X_projected[:,0]
    x2 = X_projected[:,1]

    fig=plt.figure()
    plt.scatter(x1,x2,c=y, alpha=0.8, cmap="viridis")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()



with tab2:   
    st.header("Classifier wise")
    
    data=datasets.load_wine()
    X = data.data
    y = data.target

        
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
            "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
            "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1)

    model_cols = []
    df=pd.DataFrame(columns=model_cols)
    index=0

    for name, clf in zip(names, classifiers):
        clf.fit(x_train,y_train)
        df.loc[index,'Classifiers'] = name
        df.loc[index,'Train Accuracy'] = clf.score(x_train,y_train)
        df.loc[index,'Test Accuracy'] = clf.score(x_test,y_test)
        df.loc[index,'Precision'] = precision_score(y_test,clf.predict(x_test),average='weighted')
        df.loc[index,'Recall'] = recall_score(y_test,clf.predict(x_test),average='weighted')
        df.loc[index,'F1 Score'] = f1_score(y_test,clf.predict(x_test),average='weighted')
        index+=1




    st.write("**Wine Dataset**")
    parameter_name = st.selectbox("Select Performance Metric",("Train Accuracy", "Test Accuracy", "Precision", "Recall","F1 Score"))

    st.write(f"**About {parameter_name} Performance Metric:**")
    if parameter_name == "Train Accuracy":
        st.write("Train accuracy is a measure of how well a model is able to predict the outcomes of the training data. It is calculated by comparing the model's predicted labels with the actual labels in the training set. A higher train accuracy indicates that the model is fitting well to the training data.")
    elif parameter_name == "Test Accuracy":
        st.write("Test accuracy is a measure of how well a model is able to predict the outcomes of new data that it has not seen before. It is calculated by comparing the model's predicted labels with the actual labels in the test set. A higher test accuracy indicates that the model is able to generalize well to new data. ")
    elif parameter_name == "Precision":
        st.write("Precision is a measure of the model's ability to identify positive instances correctly. It is calculated by dividing the number of true positive predictions by the total number of positive predictions made by the model. The mathematical formula for precision is precision = TP / (TP + FP), where TP is the number of true positives and FP is the number of false positives. ")
    elif parameter_name == "Recall":
        st.write("Recall is an evaluation metric in machine learning that measures the ability of a model to identify all positive instances. It is calculated by dividing the number of true positive predictions by the total number of actual positive instances in the data. The mathematical formula for recall is recall = TP / (TP + FN), where TP is the number of true positives and FN is the number of false negatives. ")
    elif parameter_name == "F1 Score":
        st.write("F1 score is an evaluation metric that combines precision and recall into a single score. It is calculated as the harmonic mean of precision and recall, and is useful when precision and recall are both important. The mathematical formula for F1 score is F1 = 2 * (precision * recall) / (precision + recall), where precision and recall are calculated as previously explained.")

    st.write("")

    if parameter_name=="Train Accuracy":
        sns.barplot(x='Classifiers',y='Train Accuracy', data=df, palette='viridis',
            edgecolor=sns.color_palette('dark',7))
        plt.xticks(rotation=90)
        plt.title('Model Train Accuracy Comparision')
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()


    if parameter_name=="Test Accuracy":
        sns.barplot(x='Classifiers',y='Test Accuracy', data=df, palette='viridis',
                    edgecolor=sns.color_palette('dark',7))
        plt.xticks(rotation=90)
        plt.title('Model Test Accuracy Comparision')
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    if parameter_name=="Precision":
        sns.barplot(x='Classifiers',y='Precision', data=df, palette='viridis',
            edgecolor=sns.color_palette('dark',7))
        plt.xticks(rotation=90)
        plt.title('Model Precision Comparision')
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()    

    if parameter_name=="Recall":
        sns.barplot(x='Classifiers',y='Recall', data=df, palette='viridis',
            edgecolor=sns.color_palette('dark',7))
        plt.xticks(rotation=90)
        plt.title('Model Recall Comparision')
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    if parameter_name=="F1 Score":
        sns.barplot(x='Classifiers',y='F1 Score', data=df, palette='viridis',
            edgecolor=sns.color_palette('dark',7))
        plt.xticks(rotation=90)
        plt.title('Model F1 Score Comparision')
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    
    # with st.sidebar: