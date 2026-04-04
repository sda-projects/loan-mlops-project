import streamlit as st
import pandas as pd
import plotly.express as px 

# Chargement des données
df = pd.read_csv("data/Loan_Data.csv")

st.title("Système d'Aide à la Décision et de Prédiction du Risque de Crédit")

# Calcul du taux de défaut
total_clients = len(df)
nb_defauts = df['default'].sum()
taux_defaut = (nb_defauts / total_clients) * 100

# Affichage des métriques en colonnes
col1, col2, col3 = st.columns(3)
col1.metric("Total Clients", total_clients)
col2.metric("Nombre de Défauts de crédit", nb_defauts)
col3.metric("Taux de Défaut Global", f"{taux_defaut:.2f}%")

st.divider() 

#Analyse des revenus des clients
st.header("Analyse des revenus des clients")

# Création de deux colonnes : une pour le Boxplot, une pour l'Histogramme
col_box, col_hist = st.columns([1, 2])

with col_box:
    # Le Boxplot permet de visualiser les valeurs extrêmes
    fig_box = px.box(df, 
                     y="income", 
                     title="Dispersion des revenus",
                     points="all", 
                     color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig_box, use_container_width=True)

with col_hist:
    
    fig_dist = px.histogram(df, 
                            x="income", 
                            nbins=50, 
                            title="Histogramme : Fréquence des Revenus",
                            color_discrete_sequence=['#636EFA'],
                            marginal="rug") 
    fig_dist.update_layout(bargap=0.1)
    st.plotly_chart(fig_dist, use_container_width=True)

#Analyse des montants des prêts
st.subheader("Analyse des montants des prêts")

col_loan1, col_loan2 = st.columns([1, 2])

with col_loan1:
    # Boxplot pour les montants de prêts
    fig_box_loan = px.box(df, 
                          y="loan_amt_outstanding", 
                          title="Dispersion des montants de prêts",
                          color_discrete_sequence=['#00CC96'])
    st.plotly_chart(fig_box_loan, use_container_width=True)

with col_loan2:
    # Histogramme pour la distribution des montants
    fig_hist_loan = px.histogram(df, 
                                 x="loan_amt_outstanding", 
                                 nbins=50, 
                                 title="Distribution des Montants de Prêts",
                                 color_discrete_sequence=['#00CC96'],
                                 marginal="rug")
    fig_hist_loan.update_layout(bargap=0.1)
    st.plotly_chart(fig_hist_loan, use_container_width=True)


st.divider() # Séparation avant l'analyse FICO

st.subheader("Relation entre le Score FICO et le Risque de Défaut")

fig_fico = px.histogram(df, x="fico_score", color="default", 
                   barmode="overlay", 
                   title="Distribution des scores FICO selon le statut de défaut",
                   color_discrete_map={0: "green", 1: "red"})
st.plotly_chart(fig_fico)

st.subheader("Analyse de l'Endettement par Revenu")

fig_scatter = px.scatter(df, x="income", y="total_debt_outstanding", 
                         color="default",
                         hover_data=['years_employed'],
                         title="Dispersion Revenu / Dette")
st.plotly_chart(fig_scatter)

# Création des onglets
tab1, tab2 = st.tabs(["🤖 Simulateur de Crédit", "📈 Performance Modèle"])

with tab1:
    st.header("Simulateur de crédit en temps réel")
    # On peut ajouter par exemple ici le code pour un formulaire de saisie et simulation de crédit
    ...

with tab2:
    st.header("Suivi MLflow")
    st.info("Cette section présente les métriques de tracking enregistrées lors des expérimentations.")
    # Afficher les images de notre interface MLflow
    # st.image("capture_mlflow.png")