import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


model  = joblib.load("student_model.pkl")
df_raw = pd.read_csv("Student_Performance.csv")
df     = df_raw.copy()
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})


st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


st.title("🎓 Student Performance Index Predictor")
st.markdown("Predict a student's **Performance Index** using an **Ensemble ML Model (Random Forest)**.")
st.divider()


st.sidebar.title("📥 Enter Student Details")

hours_studied    = st.sidebar.number_input("📚 Hours Studied (per day)",      min_value=1, max_value=9,  value=5)
previous_scores  = st.sidebar.number_input("📝 Previous Scores (0–100)",      min_value=40, max_value=99, value=75)
extracurricular  = st.sidebar.selectbox("🏃 Extracurricular Activities",      ["Yes", "No"])
sleep_hours      = st.sidebar.number_input("😴 Sleep Hours (per night)",       min_value=4, max_value=9,  value=7)
sample_papers    = st.sidebar.number_input("📄 Sample Papers Practiced",       min_value=1, max_value=9,  value=3)

extra_encoded = 1 if extracurricular == "Yes" else 0

predict_btn = st.sidebar.button("🎯 Predict Performance", use_container_width=True)


if predict_btn:
    input_data = np.array([[hours_studied, previous_scores,
                            extra_encoded, sleep_hours, sample_papers]])
    prediction = round(float(model.predict(input_data)[0]), 2)
    prediction = max(0, min(100, prediction))

    st.subheader("🔮 Prediction Result")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🎯 Performance Index", f"{prediction} / 100")
    c2.metric("📚 Hours Studied",     hours_studied)
    c3.metric("📝 Previous Scores",   previous_scores)
    c4.metric("😴 Sleep Hours",       sleep_hours)
    c5.metric("📄 Papers Practiced",  sample_papers)

    if prediction >= 75:
        st.success("🏆 Excellent performance expected! Keep it up!")
        st.balloons()
    elif prediction >= 50:
        st.info("👍 Good performance. A bit more effort will help!")
    else:
        st.warning("⚠️ Needs improvement. Try studying more hours!")

    st.divider()


st.subheader("📊 Dataset Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🔥 Correlation Heatmap**")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax1)
    st.pyplot(fig1)

with col2:
    st.markdown("**📈 Hours Studied vs Performance Index**")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(df_raw['Hours Studied'], df_raw['Performance Index'],
                alpha=0.3, color='steelblue')
    ax2.set_xlabel("Hours Studied")
    ax2.set_ylabel("Performance Index")
    ax2.set_title("Study Hours vs Performance")
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    st.markdown("**📊 Performance Index Distribution**")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.hist(df_raw['Performance Index'], bins=30,
             color='orange', edgecolor='black')
    ax3.set_title("Performance Index Distribution")
    st.pyplot(fig3)

with col4:
    st.markdown("**🏃 Extracurricular vs Avg Performance**")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    df_raw.groupby('Extracurricular Activities')['Performance Index'] \
          .mean().plot(kind='bar', ax=ax4,
                       color=['salmon', 'lightgreen'], edgecolor='black')
    ax4.set_title("Avg Performance by Extracurricular")
    ax4.set_ylabel("Avg Performance Index")
    ax4.set_xlabel("")
    plt.xticks(rotation=0)
    st.pyplot(fig4)

st.divider()


st.subheader("📋 Dataset Preview")
st.dataframe(df_raw.head(10), use_container_width=True)