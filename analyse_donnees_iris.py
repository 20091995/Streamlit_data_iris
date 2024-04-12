import pandas as pd
import numpy as np
import os
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import streamlit as st
import matplotlib as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import chart_studio.plotly as py
from plotly.subplots import make_subplots
from functools import partial, reduce
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from datetime import datetime
from PIL import Image
#Package de Machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, metrics
#from sklearn.model_selection import trai_test_split
from sklearn import svm

pd.set_option('display.max_columns', None) # Pour afficher toutes les colonnes

## Gestion des warnings
import warnings
warnings.filterwarnings('ignore')   
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
print("No Warning Shown")
pd.set_option('display.max_columns', None) # Pour afficher toutes les colonnes
############################################################################################################################################################################################################################################################################
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: lightgrey;
        primaryColor="green";
        textColor="red";
        font="sans serif";
        font="sans serif"
    }
</style>
""", unsafe_allow_html=True)

#image = Image.open(File+"\img.webp") 
#st.sidebar.image(image,use_column_width=True,width=0.5) #POUR LA PARTIE DE GAUCHE
#st.image(image,use_column_width=True,width=0.0001) #POUR LA PARTIE DE DROITE

############################################################################################################################################################################################################################################################################
Title_html = """
        <html>
        <style>
            .title h1{
              user-select: none;
              font-size: 70px;
              color: magenta;
              background: repeating-linear-gradient(-45deg, red 0%, yellow 7.14%, rgb(0,255,0) 14.28%, rgb(0,255,255) 21.4%, cyan 28.56%, blue 35.7%, magenta 42.84%, red 50%);
              background-size: 600vw 600vw;
              -webkit-text-fill-color: transparent;
              -webkit-background-clip: text;
              animation: slide 5s linear infinite forwards;
            }
            @keyframes slide {
              0%{
                background-position-x: 0%;
              }
              100%{
                background-position-x: 600vw;
              }
            }
        </style> 
        <div class="title"> <h1 align="center"> Analyse statistique des Données d'iris <br><br></h1> </div>
        </html>
        """
#st.markdown(Title_html, unsafe_allow_html=True) #Title rendering
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title(" :black[Analyse statistique des Données d'iris]")

#Choix de la visualisation de la base 
st.sidebar.title("Consultation des données d'iris")
checkbox=st.sidebar.checkbox("Affichage data iris")

#Conception de la base de données
def data_iris():
    df=datasets.load_iris()
    df1=pd.DataFrame(df.data,columns=['sepal width','sepal length','petal length','petal width'])
    df2=pd.DataFrame(df.target,columns=['fleur'])
    df3=pd.concat([df1,df2],axis=1)
    Nom_fleur=[]*len(df3.fleur)
    for i in range(len(df3.fleur)):
        if (df3["fleur"][i]==0):
            Nom_fleur.append("Setosa")
        elif (df3["fleur"][i]==1):
            Nom_fleur.append("Versicolor")
        else:
            Nom_fleur.append("Virgina")
    df3["Nom_fleur"]=Nom_fleur
    return df3

st.markdown("<h1 style='text-align: center; color: blue;'> Consultation de la base de données  </h1>", 
            unsafe_allow_html=True)

data=data_iris()
print(checkbox)
if checkbox :
   cm = sns.light_palette("green", as_cmap=True)
   st.dataframe(data=data.style.set_table_attributes('class="center"').background_gradient(cmap=cm))

# Analyse par espéce
st.sidebar.title("Choix d'analyse")
#st.sidebar.subheader("Analyse par espéce")

def Analyse_data_iris():
    Choix0 = st.sidebar.radio(" :blue[Veuillez choisir] : ", ("Analyse par espéce", "Analyse globale", "Prédiction d'espéce"))
    if (Choix0=="Analyse par espéce"):
        st.sidebar.title('Analyse par espèce')
        Choix1 = st.sidebar.radio(" :blue[Veuillez préciser l'espéce] : ", ("Setosa", "Versicolor", "Virgina"))
        # Analyse descriptive 
        def description(Choix1):
            data0=data[data["Nom_fleur"]==Choix1]
            data1=data0.drop(columns=["fleur","Nom_fleur"])
            val=data1.describe()
            val=pd.DataFrame(val)
            val=val.T
            return val, data0
        
        #Histogramme des longueurs
        def histogramme1(var1,var2):
            fig = go.Figure(data=go.Scatter(x=description(Choix1)[1][var1],y=description(Choix1)[1][var2],
            mode='markers',marker_color=description(Choix1)[1][var2]))
            fig.update_layout(title="",font_family="San Serif",bargap=0.1,titlefont={'size': 30},
                paper_bgcolor='lime',plot_bgcolor='lime',legend=dict(orientation="v", y=1, yanchor="top", x=1.250, xanchor="right",))
            fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide',width=400,height=400)
            fig.update_xaxes(title=var1,title_font=dict(size=20, family='Courier', color='black'),tickfont=dict(color="black",size=10))
            fig.update_yaxes(title=var2,title_font=dict(size=20, family='Courier', color='black'),tickfont=dict(color="black",size=10))
            return fig
    
        def histograme1_2(Choix1):
            df=description(Choix1)[1][["sepal width","sepal length","petal length","petal width"]]
            fig1=ff.create_scatterplotmatrix(df,diag='histogram', index ='petal length', 
            colormap = ['rgb(100, 150, 255)' ,  '#F0963C' ,  'rgb(51, 255, 153)' ], 
            colormap_type = 'seq', width=800, height=600 ) 
            ########################################################################################################
            fig2 = px.line(description(Choix1)[1], y=["sepal width", "sepal length", "petal length","petal width"],
             color_discrete_map={"sepal width":"blue", "sepal length":"black","petal width":"magenta","petal length":"red"},)
            fig2.update_layout(title="La variation des lon/lar",font_family="San Serif",bargap=0.1,titlefont={'size': 30},
                        paper_bgcolor='lime',plot_bgcolor='lime',legend=dict(orientation="v", y=0.7, yanchor="top", x=1.250, xanchor="center",))
            fig2.update_layout(uniformtext_minsize=10, uniformtext_mode='hide',width=760,height=400)
            fig2.update_xaxes(title="Index",title_font=dict(size=20, family='Courier', color='black'),tickfont=dict(color="black",size=10))
            fig2.update_yaxes(title="Longueur/largeur",title_font=dict(size=20, family='Courier', color='black'),tickfont=dict(color="black",size=10))
            return fig1, fig2
        
        #Matrice de corrélation
        def matrice_corr(Choix1):
            df=data[data["Nom_fleur"]==Choix1]
            corrs=df[["sepal width","sepal length","petal length","petal width"]].corr()
            figure = ff.create_annotated_heatmap(
                z=corrs.values,x=list(corrs.columns),y=list(corrs.index),colorscale='Earth',
                annotation_text=corrs.round(2).values,showscale=True, reversescale=True)
            return figure
        
        #Fonction pour définir les colonnes :
        def colonnes(var1,var2,var3,var4):
            col1, col2=st.columns(2)
            with col1 :
                st.plotly_chart(histogramme1(var1,var2))
                st.set_option('deprecation.showPyplotGlobalUse', False)
            with col2 :
                st.plotly_chart(histogramme1(var1,var4))
                st.set_option('deprecation.showPyplotGlobalUse', False)
        

        if (Choix1=="Setosa"):
            st.markdown("<h1 style='text-align: center; color: blue;'> Dashboard des espéces Setosa </h1>", unsafe_allow_html=True)
            st.write('''Cette section consiste à faire une analyse elementaire des longeures en fonction des largeurs
                     des sépales et des sépales''')
            
            print(checkbox)
            if checkbox :
                st.dataframe(data=description(Choix1)[0].style.set_table_attributes('class="center"').set_properties(**{'background-color': 'lightcyan','color': 'black'}))
            
            st.plotly_chart(histograme1_2(Choix1)[1])
            st.set_option('deprecation.showPyplotGlobalUse', False)

            colonnes("sepal width","sepal length","sepal width","petal length")
            colonnes("petal width","petal length","sepal width","petal width")
            
            st.write("Dasboard des longueurs et des largeurs des sepales et la largeur des petales en fonction des longueurs des petales")
            st.plotly_chart(histograme1_2(Choix1)[0])
            st.set_option('deprecation.showPyplotGlobalUse', False)

            st.markdown("<h1 style='text-align: center; color: blue;'> La corrélation entre les longueurs et les largeurs </h1>", unsafe_allow_html=True)
            st.plotly_chart(matrice_corr(Choix1))
            st.set_option('deprecation.showPyplotGlobalUse', False)
           
            
        elif (Choix1=="Versicolor"):
            st.markdown("<h1 style='text-align: center; color: blue;'> Dashboard des espéces Versicolor </h1>", unsafe_allow_html=True)
            st.write('''Cette section consiste à une analyse elementaire des longeures en fonction des largeurs
                     des sépales et des sépales''')
            
            print(checkbox)
            if checkbox :
                st.dataframe(data=description(Choix1)[0].style.set_properties(**{'background-color': 'lightcyan','color': 'black'}))
            
            st.plotly_chart(histograme1_2(Choix1)[1])
            st.set_option('deprecation.showPyplotGlobalUse', False)

            colonnes("sepal width","sepal length","sepal width","petal length")
            colonnes("sepal width","petal width","sepal length","petal length")
            
            st.write("Dasboard des longueurs et des largeurs des sepales et la largeur des petales en fonction des longueurs des petales")
            st.plotly_chart(histograme1_2(Choix1)[0])
            st.set_option('deprecation.showPyplotGlobalUse', False)

            st.markdown("<h1 style='text-align: center; color: blue;'> La corrélation entre les longueurs et les largeurs </h1>", unsafe_allow_html=True)
            st.plotly_chart(matrice_corr(Choix1))
            st.set_option('deprecation.showPyplotGlobalUse', False)

        else:
            st.markdown("<h1 style='text-align: center; color: blue;'> Dashboard des espéces Virgina </h1>", unsafe_allow_html=True)
            st.write('''Cette section consiste à une analyse elementaire des longeures en fonction des largeurs
                     des sépales et des sépales''')
            
            print(checkbox)
            if checkbox :
                st.dataframe(data=description(Choix1)[0].style.set_properties(**{'background-color': 'lightcyan','color': 'black'}))
            
            st.plotly_chart(histograme1_2(Choix1)[1])
            st.set_option('deprecation.showPyplotGlobalUse', False)
            colonnes("sepal width","sepal length","sepal width","petal length")
            colonnes("sepal width","petal width","sepal length","petal length")
            
            st.write("Dasboard des longueurs et des largeurs des sepales et la largeur des petales en fonction des longueurs des petales")
            st.plotly_chart(histograme1_2(Choix1)[0])
            st.set_option('deprecation.showPyplotGlobalUse', False)

            st.markdown("<h1 style='text-align: center; color: blue;'> La corrélation entre les longueurs et les largeurs </h1>", unsafe_allow_html=True)
            st.plotly_chart(matrice_corr(Choix1))
            st.set_option('deprecation.showPyplotGlobalUse', False)
   
    elif(Choix0=="Analyse globale"):
        liste_des_colonnes=list(data_iris().drop(columns=["fleur","Nom_fleur"]).columns)
        st.sidebar.title('Analyse Global')
        Liste=st.sidebar.multiselect(label=" :blue[Veuillez choisir votre composante]", options=liste_des_colonnes)
        st.markdown("<h1 style='text-align: center; color: blue;'> Dashboard des compsantes </h1>", unsafe_allow_html=True)
        
        def histograme4():
            df=data_iris().drop(columns=["fleur"])
            fig1=ff.create_scatterplotmatrix(df, diag = 'histogram', index='Nom_fleur') 
            fig1.update_layout(title="histogramme global",font_family="San Serif",bargap=0.1,titlefont={'size': 30},
                paper_bgcolor='black',plot_bgcolor='black',legend=dict(orientation="v", y=1, yanchor="top", x=1.250, xanchor="right",))
            fig1.update_layout(uniformtext_minsize=10, uniformtext_mode='hide',width=800,height=800)
            ######################################################################################################
            fig2= px.scatter_3d(df, x='sepal length', y='sepal width', z='petal width',
              color='Nom_fleur', size='petal length', size_max=18,symbol='Nom_fleur', opacity=1)
            fig2.update_layout(title="En hauteur la largeur des petales",font_family="San Serif",bargap=0.1,titlefont={'size': 30},
                paper_bgcolor='black',plot_bgcolor='black',legend=dict(orientation="v", y=1, yanchor="top", x=1.250, xanchor="right",))
            fig2.update_layout(uniformtext_minsize=10, uniformtext_mode='hide',width=800,height=600)
            return fig1,fig2
        
        def stat():
            df=data_iris().drop(columns=["fleur"])
            df_min=df.groupby(["Nom_fleur"]).min().reset_index()
            df_max=df.groupby(["Nom_fleur"]).max().reset_index()
            df_mean=df.groupby(["Nom_fleur"]).mean().reset_index()
            df_std=df.groupby(["Nom_fleur"]).std().reset_index()
            return df_min,df_max,df_mean,df_std
        
        def nom_stat(titre1, titre2):
            col1, col2=st.columns(2)
            with col1 :
                st.write(titre1)
                print(checkbox)
                if checkbox :
                    st.dataframe(data=stat()[0].style.set_properties(**{'background-color': 'lightcyan','color': 'black'}))
            with col2 :
                st.write(titre2)
                print(checkbox)
                if checkbox :
                    st.dataframe(data=stat()[1].style.set_properties(**{'background-color': 'lightcyan','color': 'black'}))
       
        nom_stat("Minimum","Maximum")
        nom_stat("Moyenne","Variance")
        
        def hisotgramme_dispersion(var):
            fig = px.box(data_iris(), x="Nom_fleur", y=var, color="Nom_fleur", notched=True)
            fig.update_layout(title="",font_family="San Serif",bargap=0.1,titlefont={'size': 10},
                paper_bgcolor='black',plot_bgcolor='black',legend=dict(orientation="v", y=1, yanchor="top", x=1.250, xanchor="right",))
            fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide',width=500,height=300)
            return fig
        
        def colonnes2(var1,var2):
            col1, col2=st.columns(2)
            with col1 :
                st.write("La dispersion de "+str(var1))
                st.plotly_chart(hisotgramme_dispersion(var1))
                st.set_option('deprecation.showPyplotGlobalUse', False)
            with col2 :
                st.write("La dispersion de "+str(var2))
                st.plotly_chart(hisotgramme_dispersion(var2))
                st.set_option('deprecation.showPyplotGlobalUse', False)
       
        colonnes2("sepal width","sepal length")
        colonnes2("petal length","petal width")
        
        st.plotly_chart(histograme4()[0])
        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.plotly_chart(histograme4()[1])
        st.set_option('deprecation.showPyplotGlobalUse', False)
    else :
        st.write('''Cette application prédit la catégorie de fleurs d'Iris''')
        st.sidebar.header("La paramétres d'entrée")

        def user_input():
            sepal_width=st.sidebar.slider("La largeur sepal",4.3,7.9,5.3)
            sepal_length=st.sidebar.slider("La longueur sepal",2.0,4.4,3.3)
            petal_length=st.sidebar.slider("La longueur petal",1.0,6.9,2.3)
            petal_width=st.sidebar.slider("La largeur petal",0.1,2.5,1.3)

            data={'sepal_width':sepal_width,'sepal_length':sepal_length,'petal_length':petal_length,'petal_width':petal_width}
            
            parametre_fleur=pd.DataFrame(data,index=[0])

            return parametre_fleur
        
        base_iris=user_input()

        st.subheader("On veut trouver la catégorie de fleur")
        st.write(base_iris)
        
        iris=datasets.load_iris() #.rename(columns=noms_nouveaux)
        clf=RandomForestClassifier()
        #X, y = df_iris[["sepal_width","sepal_length","petal_length","petal_width"]],df_iris["Nom_fleur"]
        clf.fit(iris.data,iris.target)
        #La prédiction des espéces
        prediction=clf.predict(base_iris)

        st.subheader("La catégorie de fleur d'Iris est")
        st.write(iris.target_names[prediction])

Analyse_data_iris()
  