import streamlit as st
import requests
import json

# =========================
# CONFIG
# =========================
AUTH_URL = "http://10.176.3.178:5080/api/Auth/token"
PREDICT_URL_BERT = "http://10.176.3.178:5080/api/External/Bert-Multi-Task-Classifier/predict"

# =========================
# AUTH
# =========================
def get_token():
    try:
        credentials = {
            "clientId": "client1_id",
            "clientSecret": "client1_secret"
        }

        headers = {"Content-Type": "application/json"}
        resp = requests.post(AUTH_URL, headers=headers, json=credentials)
        resp.raise_for_status()

        data = resp.json()
        return data.get("token")

    except Exception as e:
        st.error(f"❌ Failed to get token: {e}")
        return None


# =========================
# PAYLOAD BUILDER (BERT)
# =========================
def build_payload_bert(text):
    return {
        "texts": [
            {
                "id": "1",
                "text": text
            }
        ]
    }


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="BERT Text Classifier", layout="wide")
st.title("🧠 BERT Multi-Task Classification")

text_input = st.text_area(
    "Enter Text",
    height=350,
    placeholder="Paste your document / PDF text here..."
)

if st.button("Run BERT Classification"):
    if not text_input.strip():
        st.error("⚠️ Please provide input text.")
        st.stop()

    with st.spinner("🔐 Authenticating..."):
        token = get_token()
        if not token:
            st.stop()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    payload = build_payload_bert(text_input)

    with st.spinner("🚀 Running BERT inference..."):
        try:
            response = requests.post(
                PREDICT_URL_BERT,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            result = response.json()

            st.subheader("✅ BERT Classification Result")
            st.json(result)

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
