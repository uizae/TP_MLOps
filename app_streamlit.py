import streamlit as st
import requests

API_URL = "http://api:8000"  # nom du service Docker FastAPI (voir compose)

st.set_page_config(page_title="🫀 Maladie cardiaque", page_icon="🫀")

st.title("🫀 Prédiction de maladie cardiaque")
st.write("Frontend Streamlit connecté à l'API FastAPI (modèle RandomForest).")

st.sidebar.header("Paramètres du patient")

age = st.sidebar.number_input("Âge", 18, 100, 50)
sex = st.sidebar.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
cp = st.sidebar.selectbox("Douleur thoracique (cp)", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Tension au repos (trestbps)", 80, 220, 120)
chol = st.sidebar.number_input("Cholestérol (chol)", 100, 600, 250)
fbs = st.sidebar.selectbox("Glycémie à jeun > 120 (fbs)", [0, 1])
restecg = st.sidebar.selectbox("ECG de repos (restecg)", [0, 1, 2])
thalach = st.sidebar.number_input("Fréquence cardiaque max (thalach)", 60, 220, 160)
exang = st.sidebar.selectbox("Angine induite par l’effort (exang)", [0, 1])
oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 10.0, 2.0, 0.1)
slope = st.sidebar.selectbox("Pente ST (slope)", [0, 1, 2])
ca = st.sidebar.selectbox("Nb de vaisseaux colorés (ca)", [0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thal", [0, 1, 2, 3])

if st.button("Prédire"):
    features = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
    try:
        r = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=5)
        if r.status_code != 200:
            st.error(f"Erreur API {r.status_code}")
            st.code(r.text)
        else:
            data = r.json()
            pred = data["prediction"]
            p_no = data["proba"]["no_disease"]
            p_yes = data["proba"]["disease"]

            st.subheader("Résultat")
            if pred == 1:
                st.error(f"⚠️ Risque de maladie cardiaque ({p_yes*100:.1f} %)")
            else:
                st.success(f"✅ Pas de maladie cardiaque ({p_no*100:.1f} %)")

            st.write("Détails bruts de l’API :")
            st.json(data)
    except Exception as e:
        st.error(f"Erreur de connexion à l’API : {e}")
        st.info("En local Docker, l’API est jointe via le service `api` sur le port 8000.")
