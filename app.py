import streamlit as st
import pandas as pd
import plotly.express as px
import mlflow
import mlflow.sklearn
import os

# Configuration MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Chargement des données
df = pd.read_csv("data/Loan_Data.csv")

def get_best_run(dataset_mode):
    best_overall_run = None
    best_f2 = -1
    
    experiments = ["Loan_Default_Logistic_Regression", "Loan_Default_Random_Forest", "Loan_Default_XGBoost"]
    
    for exp_name in experiments:
        experiment = mlflow.get_experiment_by_name(exp_name)
        if not experiment:
            continue
            
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"params.feature_mode = '{dataset_mode}'",
            order_by=["metrics.val_f2 DESC"],
            max_results=1
        )
        
        if not runs.empty:
            run = runs.iloc[0]
            val_f2 = float(run["metrics.val_f2"])
            if val_f2 > best_f2:
                best_f2 = val_f2
                best_overall_run = run
                best_overall_run["model_name"] = exp_name.replace("Loan_Default_", "")
                
    return best_overall_run

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

# Analyse visuelle
st.header("Analyse exploratoire")
col_box, col_hist = st.columns([1, 2])

with col_box:
    fig_box = px.box(df, y="income", title="Dispersion des revenus", points="all", color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig_box, use_container_width=True)

with col_hist:
    fig_dist = px.histogram(df, x="income", nbins=50, title="Histogramme : Fréquence des Revenus", color_discrete_sequence=['#636EFA'], marginal="rug") 
    fig_dist.update_layout(bargap=0.1)
    st.plotly_chart(fig_dist, use_container_width=True)

# Analyse des montants des prêts
st.subheader("Analyse des montants des prêts")

col_loan1, col_loan2 = st.columns([1, 2])

with col_loan1:
    fig_box_loan = px.box(df, y="loan_amt_outstanding", title="Dispersion des montants de prêts", color_discrete_sequence=['#00CC96'])
    st.plotly_chart(fig_box_loan, use_container_width=True)

with col_loan2:
    fig_hist_loan = px.histogram(df, x="loan_amt_outstanding", nbins=50, title="Distribution des Montants de Prêts", color_discrete_sequence=['#00CC96'], marginal="rug")
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
    
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        income = col1.number_input("Revenu Annuel (income)", min_value=0, value=50000)
        loan_amt = col2.number_input("Montant du prêt (loan_amt_outstanding)", min_value=0, value=10000)
        years_emp = col2.number_input("Années d'emploi (years_employed)", min_value=0, value=5)
        fico = col1.number_input("Score FICO (fico_score)", min_value=300, max_value=850, value=700)
        
        credit_lines = 0
        total_debt = 0
        
        if dataset_mode == "full_features":
            credit_lines = col1.number_input("Lignes de crédit (credit_lines_outstanding)", min_value=0, value=2)
            total_debt = col2.number_input("Dette totale (total_debt_outstanding)", min_value=0, value=5000)
            
        submit = st.form_submit_button("Prédire le risque")

    # Calcul temps réel des features ingénieries
    eps = 1e-6
    if submit:
        # Affichage du calcul des features
        st.subheader("Features calculées en temps réel")
        st.write("Ces variables sont calculées automatiquement avant l'envoi au modèle :")
        
        features_calculated = {}
        cols = st.columns(2)
        
        if dataset_mode == "full_features":
            features_calculated = {
                "debt_to_income": total_debt / (income + eps),
                "loan_to_income": loan_amt / (income + eps),
                "loan_to_debt": loan_amt / (total_debt + eps),
                "debt_per_credit_line": total_debt / (credit_lines + eps),
                "income_per_credit_line": income / (credit_lines + eps),
                "credit_lines_per_year": credit_lines / (years_emp + 1),
                "loan_per_year_employed": loan_amt / (years_emp + 1),
                "fico_income_interaction": fico * income,
                "fico_debt_interaction": fico * (total_debt / (income + eps))
            }
        else:
            features_calculated = {
                "loan_to_income": loan_amt / (income + eps),
                "loan_per_year_employed": loan_amt / (years_emp + 1),
                "fico_income_interaction": fico * income
            }
        
        # Affichage des résultats dans des cartes stylisées
        
        # Configuration CSS pour les cartes
        st.markdown("""
        <style>
        .feature-card {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #ddd;
            margin-bottom: 10px;
        }
        .feature-name { font-weight: bold; color: #333; }
        .feature-formula { font-size: 0.85em; color: #666; font-style: italic; }
        .feature-val { font-size: 1.2em; color: #007bff; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)

        cols = st.columns(2)
        for i, (name, val) in enumerate(features_calculated.items()):
            label = f"{name}"
            formula = ""
            # Mapping pour labels et formules
            if name == "debt_to_income": 
                label = "Endettement global"
                formula = "Dette totale / Revenu"
            elif name == "loan_to_income": 
                label = "Charge du prêt"
                formula = "Prêt en cours / Revenu"
            elif name == "loan_to_debt": 
                label = "Poids du prêt actuel"
                formula = "Prêt en cours / Dette totale"
            elif name == "debt_per_credit_line": 
                label = "Dette moyenne/crédit"
                formula = "Dette totale / Lignes de crédit"
            elif name == "income_per_credit_line": 
                label = "Revenu/crédit"
                formula = "Revenu / Lignes de crédit"
            elif name == "credit_lines_per_year": 
                label = "Vitesse crédit"
                formula = "Lignes de crédit / (Années emploi + 1)"
            elif name == "loan_per_year_employed": 
                label = "Prêt/an d'emploi"
                formula = "Prêt en cours / (Années emploi + 1)"
            elif name == "fico_income_interaction": 
                label = "Interaction FICO/Revenu"
                formula = "Score FICO * Revenu"
            elif name == "fico_debt_interaction": 
                label = "Interaction FICO/Dette"
                formula = "Score FICO * Taux endettement"
            
            card_html = f"""
            <div class="feature-card">
                <div class="feature-name">{label}</div>
                <div class="feature-formula">{formula}</div>
                <div class="feature-val">{val:.4f}</div>
            </div>
            """
            cols[i % 2].markdown(card_html, unsafe_allow_html=True)

        # 1. Charger le meilleur run
        best_run = get_best_run(dataset_mode)
        if best_run is None:
            st.error("Aucun modèle entraîné trouvé pour ce mode.")
        else:
            run_id = best_run["run_id"]
            val_f2 = best_run["metrics.val_f2"]
            threshold = float(best_run["params.optimized_threshold"])
            model_name = best_run["model_name"]

            st.info(f"Modèle utilisé ({model_name}) - Run ID : {run_id} (Score F2 : {val_f2:.4f})")

            # 2. Charger le pipeline (inclut le scaler)
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

            # 3. Préparer les données pour le modèle
            features_model = features_calculated.copy()
            features_model.update({
                "income": income,
                "loan_amt_outstanding": loan_amt,
                "years_employed": years_emp,
                "fico_score": fico
            })
            
            if dataset_mode == "full_features":
                features_model.update({
                    "credit_lines_outstanding": credit_lines,
                    "total_debt_outstanding": total_debt
                })
                ordered_cols = [
                    "credit_lines_outstanding", "loan_amt_outstanding", "total_debt_outstanding", 
                    "income", "years_employed", "fico_score", "debt_to_income", 
                    "loan_to_income", "loan_to_debt", "debt_per_credit_line", 
                    "income_per_credit_line", "credit_lines_per_year", 
                    "loan_per_year_employed", "fico_income_interaction", "fico_debt_interaction"
                ]
            else:
                ordered_cols = [
                    "loan_amt_outstanding", "income", "years_employed", 
                    "fico_score", "loan_to_income", "loan_per_year_employed", 
                    "fico_income_interaction"
                ]
                
            input_df = pd.DataFrame([features_model])[ordered_cols]
            
            # 4. Prédiction
            proba = model.predict_proba(input_df)[0][1]
            is_default = proba >= threshold
            
            st.write("---")
            st.metric("Risque de défaut", "OUI" if is_default else "NON")
            st.progress(float(proba))
            st.write(f"Probabilité de défaut : {proba:.2%}")

            # 5. Interprétabilité (seulement pour la Régression Logistique)
            if model_name == "Logistic_Regression":
                st.subheader("Explication du risque (Coefficients)")
                clf = model.named_steps['clf']
                coeffs = pd.DataFrame(clf.coef_[0], index=input_df.columns, columns=["Poids"])
                
                # Impact = poids * valeur (déjà scalée par le pipeline)
                impact = coeffs["Poids"] * input_df.iloc[0]
                impact_df = impact.sort_values(ascending=False)
                
                fig_impact = px.bar(impact_df, orientation='h', title="Impact de chaque facteur sur le risque")
                st.plotly_chart(fig_impact)

with tab2:
    st.header("Suivi MLflow")
    st.info("Cette section présente les métriques de tracking enregistrées lors des expérimentations.")
