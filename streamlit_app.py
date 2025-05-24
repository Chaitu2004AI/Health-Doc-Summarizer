import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Label mapping
label_map = {
    0: "Cardiology",
    1: "Neurology",
    2: "Oncology",
    3: "Pulmonology",
    4: "Infectious Disease",
    5: "Endocrinology"
}

# Sample abstracts
sample_abstracts = {
    "Cardiology": "The patient exhibits symptoms of myocardial infarction including chest pain and elevated troponin levels.",
    "Neurology": "MRI revealed multiple sclerosis lesions in the white matter with associated symptoms of weakness and numbness.",
    "Oncology": "The tumor biopsy confirmed adenocarcinoma with metastasis to regional lymph nodes.",
    "Pulmonology": "Patient presents with chronic cough, wheezing, and shortness of breath suggestive of asthma.",
    "Infectious Disease": "Symptoms include fever, night sweats, and a productive cough indicating possible tuberculosis."
}

# Page setup
st.set_page_config(page_title="Healthcare Document Classifier", layout="wide")
st.title("🩺 Healthcare Document Classifier")

# Tabs for layout
single_tab, batch_tab = st.tabs(["🧾 Single Prediction", "📁 Batch Prediction"])

with single_tab:
    st.subheader("✍️ Enter Medical Abstract: (or choose a sample)")
    sample_choice = st.selectbox("Choose a Sample Abstract:", [""] + list(sample_abstracts.keys()))
    default_input = sample_abstracts[sample_choice] if sample_choice else ""

    user_input = st.text_area("Medical Abstract:", value=default_input, height=200)

    if st.button("🔍 Predict") and user_input.strip():
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        probabilities = model.predict_proba(input_vector)[0]

        predicted_label = label_map.get(prediction, str(prediction))
        confidence_percent = probabilities[prediction] * 100

        st.success(f"🧾 **Predicted Condition:** `{predicted_label}`")
        st.metric(label="Confidence (%)", value=f"{confidence_percent:.2f}")

        # Probability chart
        readable_labels = [label_map.get(i, str(i)) for i in range(len(probabilities))]
        prob_df = pd.DataFrame({'Condition': readable_labels, 'Probability': probabilities})
        st.subheader("📈 Prediction Confidence Across All Classes")
        st.bar_chart(prob_df.set_index("Condition"))

        # SHAP Explanation
        st.subheader("🧠 SHAP Explanation (Top Features)")
        try:
            background_texts = [
                "Patient has shortness of breath and cough.",
                "History of hypertension and stroke.",
                "The patient is experiencing chest pain.",
                "Diabetic patient with blurred vision and fatigue.",
                "Individual complains of severe headache and dizziness."
            ]
            background_vectors = vectorizer.transform(background_texts)

            def predict_proba_fn(X):
                return model.predict_proba(X)

            explainer = shap.KernelExplainer(predict_proba_fn, background_vectors)
            shap_values = explainer.shap_values(input_vector, nsamples=100)

            feature_names = vectorizer.get_feature_names_out()
            input_vector_array = input_vector.toarray()

            plt.figure()
            shap.summary_plot(shap_values, input_vector_array, feature_names=feature_names, plot_type="bar", show=False)
            st.pyplot(plt.gcf())
        except Exception as e:
            st.warning(f"⚠️ SHAP explanation error: {e}")

with batch_tab:
    st.subheader("📂 Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV with 'medical_abstract' column", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "medical_abstract" not in df.columns:
            st.error("❌ CSV must contain a 'medical_abstract' column.")
        else:
            st.write("📄 Uploaded Data Preview:")
            st.dataframe(df.head())

            vectors = vectorizer.transform(df["medical_abstract"].astype(str))
            preds = model.predict(vectors)
            probas = model.predict_proba(vectors)

            df["predicted_label"] = preds
            df["predicted_condition"] = df["predicted_label"].map(label_map)
            df["prediction_confidence_%"] = [f"{100 * max(p):.2f}%" for p in probas]

            st.success("✅ Batch predictions complete!")
            st.dataframe(df[["medical_abstract", "predicted_condition", "prediction_confidence_%"]])

            st.subheader("🔢 Prediction Summary")
            st.write(df["predicted_condition"].value_counts())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; padding-top: 10px; font-size: 14px;'>"
    "Made with ❤️ by <b>Deepak</b>, <b>Chaithaniya Krishnan</b>, and <b>Blesto Derin</b>"
    "</div>",
    unsafe_allow_html=True
)

