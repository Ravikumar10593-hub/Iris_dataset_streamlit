#we need to write the streamlit code now

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, accuracy_score


def main():
    
    #Give a title
    st.title("IRIS Classifier")
    st.sidebar.title("App Sidebar")
    st.markdown("Which species are you?")
    
    
    @st.cache 
    def load_data():
        data = pd.read_csv("iris.csv")
        return data.drop('Id', axis = 1)
    
    @st.cache
    def split(df):
        x = df.drop('Species', axis = 1)
        y = pd.factorize(df.Species)[0]
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25)
        return x_train, x_test, y_train, y_test
    
    if st.sidebar.checkbox('Show Data', False):
        st.write(load_data())
    
    model_run = False
    
    df = load_data()
    
    x_train, x_test, y_train, y_test = split(df)
    
    st.sidebar.subheader('Choose Classifier')
    
    classifier = st.sidebar.selectbox('Classifier', ('RandomForest', 'DecisionTrees'))
    
    if classifier == 'RandomForest':
        n_estimators = st.sidebar.number_input('Estimators', 10, 100, step = 10, key = 'Estimator')
        
        if st.sidebar.checkbox('classify', key = 'classify'):
            st.subheader('Random Forest Classifier')
            model = RandomForestClassifier(n_estimators = n_estimators)
            model.fit(x_train, y_train)
            predic = model.predict(x_test)
            accuracy = accuracy_score(y_test, predic)
            st.write("Accuracy", accuracy)
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model , x_test, y_test , display_labels = ['Setosa', 'Versicolor', 'Verginica'])
            st.pyplot()
            
    
            model_run = True

    
            
            
        if model_run == True:
            st.subheader("Predict")
            pl = st.number_input('Petal_Length')
            pw = st.number_input('Petal_Wimodelh')
            sl = st.number_input('Sepal_Length')
            sw = st.number_input('Sepal_Wimodelh')
        
        if st.button('Predict'):
            species = model.predict([[pl,pw,sl,sw]])
            
            if species == 0:
                species = 'Iris-Setosa'
                st.succes('This species of Iris is {}'.format(species))
            elif species == 1:
                species = 'Iris-Versicolor'
                st.succes('This species of Iris is {}'.format(species))
            elif species == 2:
                species = 'Iris-Verginica'
                st.succes('This species of Iris is {}'.format(species))


    
    elif classifier == 'DecisionTrees':
        st.subheader('Decision Tree Classifier')
        
        if st.sidebar.checkbox('classify', key = 'classify'):
            st.subheader('Decision Tree Classifier')
            model = DecisionTreeClassifier()
            model.fit(x_train, y_train)
            predic = model.predict(x_test)
            accuracy = accuracy_score(y_test, predic)
            st.write("Accuracy", accuracy)
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model , x_test, y_test , display_labels = ['Setosa', 'Versicolor', 'Verginica'])
            st.pyplot()
            
            model_run = True

        if model_run == True:
            st.subheader("Predict")
            pl = st.number_input('Petal_Length')
            pw = st.number_input('Petal_Wimodelh')
            sl = st.number_input('Sepal_Length')
            sw = st.number_input('Sepal_Wimodelh')
        
        if st.button('Predict'):
            species = model.predict([[pl,pw,sl,sw]])
            
            if species == 0:
                species = 'Iris-Setosa'
                st.succes('This species of Iris is {}'.format(species))
            elif species == 1:
                species = 'Iris-Versicolor'
                st.succes('This species of Iris is {}'.format(species))
            elif species == 2:
                species = 'Iris-Verginica'
                st.succes('This species of Iris is {}'.format(species))




if __name__ == '__main__':
    main()
