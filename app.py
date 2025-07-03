import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from autots import AutoTS

st.set_page_config(page_title="AutoTS WebApp", layout="centered")
st.title("📈 Predicción automática de series temporales con AutoTS")

# 1. Subida de archivo
uploaded_file = st.file_uploader("🔼 Sube un archivo CSV con la serie temporal", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)

    # Mostrar datos
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    # Configuración del modelo
    st.sidebar.header("⚙️ Parámetros de predicción")
    forecast_length = st.sidebar.slider("Horizonte de predicción (días)", 7, 90, 30)
    ensemble_mode = st.sidebar.selectbox("Tipo de Ensemble", ["simple", "best", "auto"])
    model_list_option = st.sidebar.selectbox("Lista de modelos", ["fast", "superfast", "default", "all"])
    max_generations = st.sidebar.slider("Iteraciones (generaciones)", 1, 10, 5)

    if st.button("🚀 Ejecutar predicción"):
        with st.spinner("Entrenando modelos... esto puede tardar un poco ⏳"):
            model = AutoTS(
                forecast_length=forecast_length,
                frequency='infer',
                ensemble=ensemble_mode,
                model_list=model_list_option,
                transformer_list="fast",
                max_generations=max_generations,
                drop_most_recent=1
            )

            model = model.fit(df)
            prediction = model.predict()
            forecast_df = prediction.forecast

        st.success("✅ Modelos entrenados y predicción generada")

        # Mostrar predicciones
        st.subheader("📊 Predicción futura")
        st.line_chart(forecast_df)

        # Comparar con los datos reales si se desea
        fig, ax = plt.subplots()
        df.iloc[:, 0].plot(ax=ax, label="Histórico", color="blue")
        forecast_df.iloc[:, 0].plot(ax=ax, label="Predicción", color="orange", linestyle="--")
        plt.legend()
        st.pyplot(fig)

        # Mejor modelo y métricas
        st.subheader("🏆 Mejor modelo encontrado")
        st.markdown(f"**Modelo:** {model.best_model_name}")
        st.markdown("**Parámetros:**")
        st.json(model.best_model_params)

        st.subheader("📈 Métricas de evaluación (Top 5 modelos)")
        results_df = model.results().sort_values("Score", ascending=False)
        st.dataframe(results_df[["Model", "TransformationParameters", "Score"]].head())

else:
    st.warning("👈 Sube primero un archivo CSV con una columna de valores y un índice de fecha.")
