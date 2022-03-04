import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

st.title("StreamLit example")
st.write("""

#### Explore different classifiers

#### Which one is the best?
""")

data_set_name = st.sidebar.selectbox("Which dataset you gonna training for?",("Iris","Breast Cancer","Wine"))

classifier_name = st.sidebar.selectbox("Select Classifier to use",("KNN","SVM","Random Forest"))

def get_dataset(data_set_name):
  if data_set_name == "Iris":
    data_set = datasets.load_iris()
  elif data_set_name == "Breast Cancer":
    data_set = datasets.load_breast_cancer()
  else:
    data_set = datasets.load_wine()
  x = data_set.data
  y = data_set.target

  return x,y

X , Y = get_dataset(data_set_name)
st.write("Shape of Dataset",X.shape)
st.write("No of classes", len(np.unique(Y)))

#Slider for Hyperparameter values

def param_search(clf_name):
  params = dict()
  
  if clf_name == "KNN":
    cl = st.sidebar.slider("K",1,15)
    params["K"] = cl
  elif clf_name == "SVM":
    cl = st.sidebar.slider("C",0.01,10.0)
    params["C"] = cl
  else:
    max_depth = st.sidebar.slider("Max-depth",2,15)
    estimators = st.sidebar.slider("n_estimators",1,1000)
    params["n_estimators"] = estimators
    params["Max-depth"] = max_depth
  return params

params = param_search(classifier_name)

#Getting classifier API

def get_classifier_api(clf_name,params):
  if clf_name == "KNN":
    clf = KNeighborsClassifier(n_neighbors=params["K"])
  elif clf_name == "SVM":
    clf = SVC(C=params["C"])
  else:
    clf = RandomForestClassifier(max_depth=params["Max-depth"],n_estimators=params["n_estimators]"],random_state = 1234)
  return clf

clf_api = get_classifier_api(classifier_name,params)

#Training the model

x_train, x_test,y_train, y_test = train_test_split(X,Y,test_size = 0.2,random_state = 1234)

clf_api.fit(x_train,y_train)

#Prediction 

y_pred = clf_api.predict(x_test)
acc = accuracy_score(y_test,y_pred)

st.write("Classifier : {}".format(classifier_name))
st.write("Accuracy = {}".format(acc))

#Plot

pca = PCA(2)  #Two dimension reduction
x_projected = pca.fit_transform(X)

x1 = x_projected[:,0]
x2 = x_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2,c=Y,alpha = 0.8 , cmap="viridis")
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.colorbar()

st.pyplot(fig)

