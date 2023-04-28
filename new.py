import pandas as pd
import numpy as np
from sklearn import datasets
import streamlit as st

data=datasets.load_wine()
X = data.data
y = data.target


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

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                                                    random_state=1)

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

import seaborn as sns
sns.barplot(x='Classifiers',y='Train Accuracy', data=df, palette='viridis',
            edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Model Train Accuracy Comparision')
plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


import seaborn as sns
sns.barplot(x='Classifiers',y='Precision', data=df, palette='viridis',
            edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Model Precision Comparision')
plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


import seaborn as sns
sns.barplot(x='Classifiers',y='Recall', data=df, palette='viridis',
            edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Model Recall Comparision')
plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


import seaborn as sns
sns.barplot(x='Classifiers',y='F1 Score', data=df, palette='viridis',
            edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Model F1 Score Comparision')
plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

