import streamlit as st
import pandas as pd
import requests
import time
import os

API_URL = os.environ.get("API_URL", "https://heart-disease-api-n9jg.onrender.com")

st.set_page_config(
    page_title="Detecteur Maladie Cardiaque",
    page_icon="🫀",
    layout="wide"
)

@st.cache_data(ttl=300)
def predict_api(features, api_url):
    time.sleep(0.3)
    response = requests.post(api_url + "/predict", json={"features": features})
    return response.json()

# En-tete
st.markdown("""
    <h1 style='text-align: center; color: #c0392b;'>Detecteur de Maladie Cardiaque</h1>
    <p style='text-align: center; color: gray; font-size: 14px;'>
        Projet MLOps - Master 1 IA | OULD SLIMANE Ouiza
    </p>
    <hr>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### Configuration")
api_url = st.sidebar.text_input("URL de l'API", API_URL)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**A propos du modele**
- Modele : Random Forest
- Dataset : Heart Disease UCI
- Accuracy : 80.4%
- F1-score : 82.4%
""")
st.sidebar.markdown("---")
st.sidebar.markdown("""
<p style='font-size: 12px; color: gray;'>
OULD SLIMANE Ouiza<br>
Master 1 - IA<br>
TP MLOps 2025-2026
</p>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Prediction manuelle", "Prediction par fichier CSV"])

with tab1:
    st.markdown("#### Renseignez les informations du patient")
    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Informations generales**")
        age = st.number_input("Age", 20, 80, 60)
        sex = st.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
        cp = st.selectbox("Type de douleur thoracique (cp)", [0, 1, 2, 3],
                          format_func=lambda x: {0: "0 - Asymptomatique", 1: "1 - Angine atypique",
                                                  2: "2 - Douleur non-angineuse", 3: "3 - Angine typique"}[x], index=1)
        trestbps = st.number_input("Tension au repos (mmHg)", 80, 220, 140)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 300)

    with col2:
        st.markdown("**Examens cliniques**")
        fbs = st.selectbox("Glycemie a jeun > 120 mg/dl", [0, 1],
                           format_func=lambda x: "Non" if x == 0 else "Oui", index=1)
        restecg = st.selectbox("ECG au repos", [0, 1, 2],
                               format_func=lambda x: {0: "0 - Normal", 1: "1 - Anomalie ST-T", 2: "2 - Hypertrophie"}[x], index=1)
        thalach = st.number_input("Frequence cardiaque maximale", 60, 220, 150)
        exang = st.selectbox("Angine induite par effort", [0, 1],
                             format_func=lambda x: "Non" if x == 0 else "Oui", index=1)

    with col3:
        st.markdown("**Parametres cardiaques**")
        oldpeak = st.number_input("Oldpeak (depression ST)", 0.0, 10.0, 2.0, 0.1)
        slope = st.selectbox("Pente du segment ST", [0, 1, 2],
                             format_func=lambda x: {0: "0 - Descendante", 1: "1 - Plate", 2: "2 - Ascendante"}[x], index=0)
        ca = st.selectbox("Nb vaisseaux colores (ca)", [0, 1, 2, 3, 4], index=1)
        thal = st.selectbox("Thalassemie (thal)", [0, 1, 2, 3],
                            format_func=lambda x: {0: "0 - Normal", 1: "1 - Defaut fixe",
                                                    2: "2 - Defaut reversible", 3: "3 - Autre"}[x], index=1)

    st.markdown("")
    col_btn, col_empty = st.columns([1, 3])
    with col_btn:
        predire = st.button("Lancer la prediction", use_container_width=True)

    if predire:
        data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope,
            "ca": ca, "thal": thal
        }

        with st.spinner("Analyse en cours..."):
            result = predict_api(data, api_url)

        pred = result["prediction"]
        p_malade = result["proba"]["disease"]
        p_sain = result["proba"]["no_disease"]
        latency = result["latency_ms"]

        st.markdown("---")
        st.markdown("#### Resultat de l'analyse")

        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            if pred == 1:
                st.error(f"Risque detecte\n\n**{p_malade*100:.1f}%** de probabilite de maladie")
            else:
                st.success(f"Pas de risque detecte\n\n**{p_sain*100:.1f}%** de probabilite d'etre sain")

        with col_res2:
            st.metric("Probabilite malade", f"{p_malade*100:.1f}%")
            st.metric("Probabilite sain", f"{p_sain*100:.1f}%")

        with col_res3:
            st.metric("Latence API", f"{latency:.1f} ms")
            st.metric("Version modele", result["model_version"])

        st.markdown("")
        with st.expander("Voir la reponse brute de l'API"):
            st.json(result)

with tab2:
    st.markdown("#### Prediction par lot depuis un fichier CSV")
    st.markdown("Le fichier doit contenir les colonnes : `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`")

    fichier_csv = st.file_uploader("Choisir un fichier CSV", type="csv")

    if fichier_csv is not None:
        df = pd.read_csv(fichier_csv)

        st.markdown("**Apercu du fichier**")
        st.write(f"{len(df)} patient(s) detecte(s)")
        st.dataframe(df.head(5), use_container_width=True)

        if st.button("Lancer les predictions", use_container_width=False):
            st.markdown("**Resultats**")
            predictions = []
            progress = st.progress(0)
            status = st.empty()

            for i, row in df.iterrows():
                status.text(f"Analyse du patient {i+1}/{len(df)}...")
                data = row.to_dict()
                resultat = predict_api(data, api_url)
                predictions.append({
                    "Patient": i+1,
                    "Prediction": "Malade" if resultat["prediction"] == 1 else "Pas malade",
                    "Confiance sain": f"{float(resultat['proba'].get('no_disease', 0))*100:.1f}%",
                    "Confiance malade": f"{float(resultat['proba'].get('disease', 0))*100:.1f}%",
                    "Latence (ms)": f"{resultat['latency_ms']:.1f}"
                })
                progress.progress((i+1)/len(df))

            status.empty()
            df_resultats = pd.DataFrame(predictions)
            st.dataframe(df_resultats, use_container_width=True)

            nb_malades = sum(1 for p in predictions if p["Prediction"] == "Malade")
            nb_sains = len(predictions) - nb_malades

            st.markdown("---")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Total patients", len(predictions))
            with col_m2:
                st.metric("Malades detectes", nb_malades)
            with col_m3:
                st.metric("Sains detectes", nb_sains)
    else:
        st.info("Selectionne un fichier CSV pour commencer")
