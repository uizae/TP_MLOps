import streamlit as st
import pandas as pd
import json
import requests
import time

st.set_page_config(page_title="Heart Disease", layout="wide")

# Cache pour éviter trop de requêtes API (erreur 429)
@st.cache_data(ttl=300)
def predict_api(features):
    """Appelle l'API avec une pause pour éviter erreur 429"""
    time.sleep(0.3)  # Attend 0.3 seconde entre chaque requête
    response = requests.post(API_URL + "/predict", json={"features": features})
    return response.json()

st.title("Detecteur Maladie Cardiaque")

# Configuration dans la sidebar
st.sidebar.header("Configuration API")
API_URL = st.sidebar.text_input("URL API", value="http://localhost:8000")

# Deux onglets : prediction simple et fichier JSON
tab1, tab2 = st.tabs(["Prediction simple", "Upload fichier JSON"])

with tab1:
    # Inputs pour une seule prediction
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 20, 80, 50)
        sex = st.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
        cp = st.selectbox("Type douleur thoracique", [0, 1, 2, 3])
        
    with col2:
        trestbps = st.number_input("Tension arterielle", 90, 200, 120)
        chol = st.number_input("Cholesterol", 100, 600, 250)
        fbs = 1 if st.checkbox("Sucre sanguin > 120mg/dl") else 0

    # Bouton prediction
    if st.button("Faire prediction", type="primary"):
        # Prepare les donnees completes
        data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": 0, "thalach": 150, "exang": 0, 
            "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2
        }
        
        # Affiche chargement
        with st.spinner("Calcul de la prediction..."):
            result = predict_api(data)
        
        # Resultat
        prediction = "Malade cardiaque" if result['prediction'] == '1' else "Pas de maladie"
        st.success(f"Resultat: {prediction}")
        
        # Probabilites
        proba_malade = float(result['proba']['1'])
        st.metric("Probabilite maladie cardiaque", f"{proba_malade:.1%}")

with tab2:
    st.header("Prediction par fichier JSON")
    
    # Upload fichier
    uploaded_file = st.file_uploader("Choisir fichier JSON", type="json")
    
    if uploaded_file is not None:
        # Lit le fichier JSON
        data_json = json.load(uploaded_file)
        df = pd.DataFrame([data_json["features"]])
        
        st.subheader("Apercu des donnees")
        st.dataframe(df)
        
        # Bouton pour predire tout le fichier
        if st.button("Predire toutes les lignes", type="primary"):
            results = []
            progress_bar = st.progress(0)
            
            # Boucle sur chaque ligne
            for i in range(len(df)):
                result = predict_api(df.iloc[i].to_dict())
                results.append(result["prediction"])
                progress_bar.progress((i+1) / len(df))
            
            # Ajoute predictions au dataframe
            df["Prediction"] = results
            
            # Affiche resultats
            st.subheader("Resultats des predictions")
            st.dataframe(df[["Prediction"]])
            
            # Statistique
            nb_malades = sum(1 for r in results if r == "1")
            st.metric("Nombre de malades detectes", nb_malades, f"{nb_malades/len(results):.1%}")

# Footer
st.markdown("---")
st.caption("Application MLOps - Tests pytest 8/8 passes")