import streamlit as st
import pandas as pd
import requests
import time
import os

API_URL = os.environ.get("API_URL", "https://heart-disease-api-n9jg.onrender.com")

st.set_page_config(layout="wide")

@st.cache_data(ttl=300)
def predict_api(features, api_url):
    time.sleep(0.3)
    response = requests.post(api_url + "/predict", json={"features": features})
    return response.json()

st.title("Detecteur Maladie Cardiaque")

st.sidebar.header("API")
api_url = st.sidebar.text_input("URL", API_URL)

tab1, tab2 = st.tabs(["1. Simple", "2. CSV"])

with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 20, 80, 60)
        sex = st.selectbox("Sexe (0=Femme, 1=Homme)", [0, 1], index=0)
        cp = st.selectbox("Type douleur thoracique (cp)", [0, 1, 2, 3], index=1)
        trestbps = st.number_input("Tension au repos (trestbps)", 80, 220, 140)
        chol = st.number_input("Cholesterol (chol)", 100, 600, 300)

    with col2:
        fbs = st.selectbox("Glycemie a jeun > 120 (fbs)", [0, 1], index=1)
        restecg = st.selectbox("ECG repos (restecg)", [0, 1, 2], index=1)
        thalach = st.number_input("Freq cardiaque max (thalach)", 60, 220, 150)
        exang = st.selectbox("Angine effort (exang)", [0, 1], index=1)

    with col3:
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 2.0, 0.1)
        slope = st.selectbox("Pente ST (slope)", [0, 1, 2], index=0)
        ca = st.selectbox("Nb vaisseaux colores (ca)", [0, 1, 2, 3, 4], index=1)
        thal = st.selectbox("Thal", [0, 1, 2, 3], index=1)

    if st.button("Predire"):
        data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope,
            "ca": ca, "thal": thal
        }
        result = predict_api(data, api_url)

        pred = result["prediction"]
        p_malade = result["proba"]["disease"]
        p_sain = result["proba"]["no_disease"]

        if pred == 1:
            st.error(f"Risque de maladie cardiaque ({p_malade*100:.1f}%)")
        else:
            st.success(f"Pas de maladie cardiaque ({p_sain*100:.1f}%)")

        st.json(result)

with tab2:
    st.header("Upload et predire CSV")
    fichier_csv = st.file_uploader("1. Choisir CSV", type="csv")

    if fichier_csv is not None:
        df = pd.read_csv(fichier_csv)
        st.subheader("2. Apercu fichier")
        st.write(f"{len(df)} lignes detectees")
        st.dataframe(df.head(3))

        if st.button("3. Predire toutes les lignes"):
            st.subheader("4. Resultats")
            predictions = []
            progress = st.progress(0)

            for i, row in df.iterrows():
                data = row.to_dict()
                resultat = predict_api(data, api_url)
                predictions.append({
                    "Ligne": i+1,
                    "Prediction": "Malade" if resultat["prediction"] == 1 else "Pas malade",
                    "Conf_Sain": f"{float(resultat['proba'].get('no_disease', 0))*100:.1f}%",
                    "Conf_Malade": f"{float(resultat['proba'].get('disease', 0))*100:.1f}%"
                })
                progress.progress((i+1)/len(df))

            df_resultats = pd.DataFrame(predictions)
            st.dataframe(df_resultats)

            nb_malades = sum(1 for p in predictions if p["Prediction"] == "Malade")
            nb_sains = len(predictions) - nb_malades
            st.metric("Patients malades detectes", f"{nb_malades}/{len(predictions)}")
            st.metric("Patients sains detectes", f"{nb_sains}/{len(predictions)}")

    else:
        st.info("Selectionne un fichier CSV d'abord")