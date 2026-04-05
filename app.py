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
    
    dataset_mode = st.radio("Sélectionner le type de modèle (features) :", ["full_features", "safe_features"])
    
    st.subheader("Entrez les informations du client")
    
    # Inputs dynamiques basés sur le mode
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        income = col1.number_input("Revenu Annuel", min_value=0, value=50000)
        loan_amt = col2.number_input("Montant du prêt", min_value=0, value=10000)
        years_emp = col1.number_input("Années d'emploi", min_value=0, value=5)
        fico = col2.number_input("Score FICO", min_value=300, max_value=850, value=700)
        
        credit_lines = None
        total_debt = None
        
        if dataset_mode == "full_features":
            credit_lines = col1.number_input("Lignes de crédit", min_value=0, value=2)
            total_debt = col2.number_input("Dette totale", min_value=0, value=5000)
            
        submit = st.form_submit_button("Prédire le risque")

    if submit:
        # Reconstitution des features pour le modèle
        eps = 1e-6
        debt_to_income = total_debt / (income + eps) if total_debt is not None else 0
        loan_to_income = loan_amt / (income + eps)
        loan_to_debt = loan_amt / (total_debt + eps) if total_debt is not None else 0
        debt_per_credit = total_debt / (credit_lines + eps) if total_debt is not None else 0
        income_per_credit = income / (credit_lines + eps) if credit_lines is not None else 0
        credit_per_year = credit_lines / (years_emp + 1) if credit_lines is not None else 0
        loan_per_year = loan_amt / (years_emp + 1)
        fico_income = fico * income
        fico_debt = fico * debt_to_income
        
        # NOTE : Ici, charger le modèle via MLflow ou un chemin local.
        # Pour cet exemple de prototype, nous affichons les données prêtes pour le modèle.
        st.write("---")
        st.success("Simulation en cours (Modèle non chargé dans ce prototype)")
        st.json({
            "income": income,
            "loan_amt_outstanding": loan_amt,
            "years_employed": years_emp,
            "fico_score": fico,
            "debt_to_income": debt_to_income,
            "loan_to_income": loan_to_income
        })

with tab2:
    st.header("Suivi MLflow")
    st.info("Cette section présente les métriques de tracking enregistrées lors des expérimentations.")
    # Afficher les images de notre interface MLflow
    # st.image("capture_mlflow.png")