import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import numpy as np

#reading the dataframe
data = pd.read_csv('averages_project_3.csv')

def displot():
    st.subheader("Histogram")
    sns.set_theme()
    fig1 = plt.figure(figsize = (6,6))
    sns.histplot(data['D'], kde = True, color = 'green')
    plt.xlabel("A")
    plt.ylabel("Frequency")
    plt.title("Distribution plot of the variable D")
    st.pyplot(fig1)

def violinplot():
    st.subheader("Violin Plot")
    sns.set_theme()
    fig2 = plt.figure(figsize = (6,6))
    sns.violinplot(data['C'], color = 'red')
    plt.xlabel("C")
    plt.title("Violin plot of the variable C")
    st.pyplot(fig2)

def plotly_hist():
    st.subheader("Plotly Histogram")
    fig3 = px.histogram(data, x = "B", marginal = 'rug')
    st.plotly_chart(fig3)

def plotly_heatmap():
    st.subheader("Plotly Heatmap")
    fig4 = px.density_heatmap(data, x="A", y="B", marginal_x="rug", marginal_y="histogram")
    st.plotly_chart(fig4)


#title for the web app
st.title("Average AI")

#we will  use a select box to navigate through datasets and predictions
navigation = st.selectbox("Select any option", ['Dataset','Analysis Dashboard','Prediction Dashboard'])

if navigation == "Dataset":
    st.header("Dataset 💽")
    #setting the dataset
    st.dataframe(data)

if navigation == "Analysis Dashboard":
    st.header("Analysis Dashboard 📈")
    graphs = st.radio("Choose a graph type",['Static Graphs', 'Interactive Graphs'])

    if graphs == 'Static Graphs':
        st.subheader(graphs)
        displot()
        violinplot()
    
    if graphs == 'Interactive Graphs':
        st.subheader(graphs)
        plotly_hist()
        plotly_heatmap()

if navigation == "Prediction Dashboard":
    st.title("Prediction Dashboard 💻")

    A = st.slider("Number1", 0.00, 999.00, 100.00)
    B = st.slider("Number 2", 0.00, 999.00, 100.00)
    C = st.slider("Number 3", 0.00, 999.00, 100.00)
    D = st.slider("Number 4", 0.00, 999.00, 100.00)

    input_array = np.array([[A,B,C,D]])

    model = pickle.load(open("average",'rb'))
    predictions = model.predict(input_array)
    st.subheader("Average: {}".format(str(predictions[0])))
    
   
