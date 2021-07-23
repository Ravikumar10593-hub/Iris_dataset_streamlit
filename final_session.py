import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,plot_confusion_matrix

def main():
    
    st.title('Iris Species Classifier')
    st.sidebar.title('App Sidebar')
    st.markdown('Which species are you?🌸')
    
    @st.cache
    def load_data():
        data = pd.read_csv('iris.csv')
        return data.drop('Id',axis=1)
    
    @st.cache
    def split(df):
        x = df.drop('Species',axis=1)
        y = pd.factorize(df.Species)[0]
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)
        return x_train,x_test,y_train,y_test
    
    if st.sidebar.checkbox('show data',False):
        st.write(load_data())
        
    model_run = False
    
    df = load_data()
    x_train,x_test,y_train,y_test = split(df)
    
    st.sidebar.subheader('Choose Classifier')
    classifier = st.sidebar.selectbox('Classifier',('RandomForest','DecisionTrees'))
    
    if classifier == 'RandomForest':
        n_estimators = st.sidebar.number_input('Estimator',10,100,step=10,key='Estimator')
        
        if st.sidebar.checkbox('Classify',key='Classify'):
            st.subheader('Random Forest Classifier')
            model = RandomForestClassifier(n_estimators=n_estimators)
            model.fit(x_train,y_train)
            predict = model.predict(x_test)
            accuracy = accuracy_score(y_test,predict)
            st.write('Accuracy',accuracy)
            st.subheader('Confusion Matrix')
            plot_confusion_matrix(model,x_test,y_test,display_labels=['Setosa','Versicolor','Virginica'])
            st.pyplot()
            
            model_run = True
            
    if model_run == True:
        st.subheader('Predict')
        pl = st.number_input('Petal_Length')
        pw = st.number_input('Petal_Width')
        sl = st.number_input('Sepal_Length')
        sw = st.number_input('Sepal_Width')
        
        if st.button('Predict'):
            species = model.predict([[pl,pw,sl,sw]])
            
            if species == 0:
                species = 'Iris-Setosa'
                st.success('The species of iris is {}'.format(species))
            elif result == 1:
                species = 'Iris-Versicolor'
                st.success('The species of iris is {}'.format(species))
            elif result == 2:
                species = 'Iris-Virginica'
                st.success('The species of iris is {}'.format(species))

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
        st.subheader('Predict')
        pl = st.number_input('Petal_Length')
        pw = st.number_input('Petal_Width')
        sl = st.number_input('Sepal_Length')
        sw = st.number_input('Sepal_Width')
        
        if st.button('Predict'):
            species = model.predict([[pl,pw,sl,sw]])
            
            if species == 0:
                species = 'Iris-Setosa'
                st.success('The species of iris is {}'.format(species))
            elif result == 1:
                species = 'Iris-Versicolor'
                st.success('The species of iris is {}'.format(species))
            elif result == 2:
                species = 'Iris-Virginica'
                st.success('The species of iris is {}'.format(species))
    
        
    
if __name__ == '__main__':
    main()
