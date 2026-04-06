import streamlit as st
import pandas as pd
import plotly.express as px
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import os
from pathlib import Path

# Configuration MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
MLFLOW_CLIENT = MlflowClient()

FEATURE_COLUMNS = {
    "full_features": [
        "credit_lines_outstanding",
        "loan_amt_outstanding",
        "total_debt_outstanding",
        "income",
        "years_employed",
        "fico_score",
    ],
    "safe_features": [
        "loan_amt_outstanding",
        "income",
        "years_employed",
        "fico_score",
    ],
}

SIMULATOR_MODEL_PRIORITY = [
    "Loan_Default_Logistic_Regression_Local",
    "Loan_Default_Random_Forest_Local",
    "Loan_Default_XGBoost_Local",
]

# Chargement des données
df = pd.read_csv("data/Loan_Data.csv")


def model_feature_names(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    for step in model.named_steps.values():
        if hasattr(step, "feature_names_in_"):
            return list(step.feature_names_in_)

    return None


def get_model_artifact_path(run_id):
    artifact_uri = MLFLOW_CLIENT.get_run(run_id).info.artifact_uri
    if artifact_uri.startswith("file:"):
        artifact_uri = artifact_uri.removeprefix("file:")
    return str(Path(artifact_uri) / "model")


def get_logistic_pipeline(model):
    if hasattr(model, "named_steps"):
        return model

    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        calibrated_estimator = model.calibrated_classifiers_[0].estimator
        if hasattr(calibrated_estimator, "named_steps"):
            return calibrated_estimator

    if hasattr(model, "estimator") and hasattr(model.estimator, "named_steps"):
        return model.estimator

    return None

def get_best_run(dataset_mode, experiment_priority=None):
    expected_columns = FEATURE_COLUMNS[dataset_mode]
    experiments = experiment_priority or [
        "Loan_Default_Logistic_Regression_Local",
        "Loan_Default_Random_Forest_Local",
        "Loan_Default_XGBoost_Local",
    ]

    for exp_name in experiments:
        experiment = mlflow.get_experiment_by_name(exp_name)
        if not experiment:
            continue
            
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                "attributes.status = 'FINISHED' "
                f"and params.feature_mode = '{dataset_mode}'"
            ),
            order_by=["metrics.val_f2 DESC"],
            max_results=20
        )
        
        for _, run in runs.iterrows():
            try:
                model = mlflow.sklearn.load_model(get_model_artifact_path(run["run_id"]))
            except MlflowException:
                continue

            feature_names = model_feature_names(model)

            if feature_names is not None and feature_names != expected_columns:
                continue

            run["model_name"] = exp_name.replace("Loan_Default_", "")
            return run

    return None

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
    st.plotly_chart(fig_box, width='stretch')

with col_hist:
    fig_dist = px.histogram(df, x="income", nbins=50, title="Histogramme : Fréquence des Revenus", color_discrete_sequence=['#636EFA'], marginal="rug") 
    fig_dist.update_layout(bargap=0.1)
    st.plotly_chart(fig_dist, width='stretch')

# Analyse des montants des prêts
st.subheader("Analyse des montants des prêts")

col_loan1, col_loan2 = st.columns([1, 2])

with col_loan1:
    fig_box_loan = px.box(df, y="loan_amt_outstanding", title="Dispersion des montants de prêts", color_discrete_sequence=['#00CC96'])
    st.plotly_chart(fig_box_loan, width='stretch')

with col_loan2:
    fig_hist_loan = px.histogram(df, x="loan_amt_outstanding", nbins=50, title="Distribution des Montants de Prêts", color_discrete_sequence=['#00CC96'], marginal="rug")
    fig_hist_loan.update_layout(bargap=0.1)
    st.plotly_chart(fig_hist_loan, width='stretch')


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

    # Prédiction en temps réel
    if submit:
        # 1. Section : Résumé des inputs
        st.subheader("Données saisies")
        input_data = {
            "income": income,
            "loan_amt_outstanding": loan_amt,
            "years_employed": years_emp,
            "fico_score": fico
        }
        if dataset_mode == "full_features":
            input_data.update({
                "credit_lines_outstanding": credit_lines,
                "total_debt_outstanding": total_debt
            })
        
        st.write("Variables utilisées :", input_data)
        st.divider()

        # 2. Section : Prédiction et Analyse
        with st.container():
            st.subheader("Résultat de l'analyse")
            
            # 1. Charger le meilleur run
            best_run = get_best_run(
                dataset_mode,
                experiment_priority=SIMULATOR_MODEL_PRIORITY,
            )
            if best_run is None:
                st.error("Aucun modèle entraîné trouvé pour ce mode.")
            else:
                run_id = best_run["run_id"]
                val_f2 = best_run["metrics.val_f2"]
                threshold = float(best_run["params.optimized_threshold"])
                model_name = best_run["model_name"]

                st.caption(f"Modèle : {model_name} | ID : {run_id} | F2-Score : {val_f2:.4f}")

                st.caption(f"Seuil de decision : {threshold:.1%}")

                # 2. Charger le pipeline
                model = mlflow.sklearn.load_model(get_model_artifact_path(run_id))

                # 3. Préparer les données pour le modèle
                # On force la liste des colonnes dans l'ordre exact attendu par le modèle
                cols_order = FEATURE_COLUMNS[dataset_mode]
                
                # Construction du DataFrame avec l'ordre forcé
                input_df = pd.DataFrame([input_data])[cols_order]
                
                # 4. Prédiction
                proba = model.predict_proba(input_df)[0][1]
                is_default = proba >= threshold
                
                status_text = "OUI" if is_default else "NON"
                status_color = "red" if is_default else "green"

                # Conteneur des résultats
                with st.container(border=True):
                    st.markdown(f"#### Risque de défaut : <span style='color:{status_color}'>{status_text}</span>", unsafe_allow_html=True)
                    st.caption(f"Probabilité : {proba:.2%}")
                    st.progress(float(proba))

                    # 5. Interprétabilité (seulement pour la Régression Logistique)
                    logistic_pipeline = get_logistic_pipeline(model)
                    if model_name.startswith("Logistic_Regression") and logistic_pipeline is not None:
                        st.write("##### Analyse des facteurs d'influence")
                        
                        clf = logistic_pipeline.named_steps['clf']
                        scaled_input = logistic_pipeline.named_steps['scaler'].transform(input_df)
                        coeffs = pd.DataFrame(clf.coef_[0], index=input_df.columns, columns=["Poids"])
                        
                        impact = coeffs["Poids"].values * scaled_input[0]
                        impact_df = pd.DataFrame(impact, index=input_df.columns, columns=["Impact"])
                        impact_df = impact_df.sort_values(by="Impact", ascending=False)
                        
                        fig_impact = px.bar(impact_df, orientation='h', 
                                            color=impact_df["Impact"] > 0, 
                                            color_discrete_map={True: '#FF4B4B', False: '#2ECC71'})
                        fig_impact.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=20), height=300)
                        st.plotly_chart(fig_impact, width='stretch')
                    elif model_name.startswith("Logistic_Regression"):
                        st.caption("Analyse des facteurs indisponible sur la version calibrÃ©e du modÃ¨le.")

with tab2:
    st.header("Suivi MLflow")
    st.info("Cette section présente les métriques de tracking enregistrées lors des expérimentations.")
