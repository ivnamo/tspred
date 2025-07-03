import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from autots import AutoTS
import time

st.set_page_config(page_title="AutoTS WebApp", layout="centered")
st.title("📈 Predicción automática de series temporales con AutoTS")

# Subida de archivo
uploaded_file = st.file_uploader("🔼 Sube un archivo CSV con la serie temporal", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    # Configuración
    st.sidebar.header("⚙️ Parámetros")
    forecast_length = st.sidebar.slider("Horizonte de predicción", 7, 90, 30)
    ensemble_mode = st.sidebar.selectbox("Tipo de Ensemble", ["simple", "best", "auto"])
    model_list_option = st.sidebar.selectbox("Lista de modelos", ["superfast", "fast", "default", "all"])
    max_generations = st.sidebar.slider("Generaciones (iteraciones)", 1, 10, 5)

    if st.button("🚀 Ejecutar predicción"):
        # Estimar número total de modelos (aproximado)
        estimated_models = {
            "superfast": 30,
            "fast": 60,
            "default": 100,
            "all": 150
        }
        total_models = estimated_models.get(model_list_option, 50) * max_generations

        st.info(f"🧮 Estimando alrededor de {total_models} modelos por evaluar...")

        # Simulación de progreso estimado (para feedback visual)
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(40):  # simulación parcial (puede ajustarse)
            time.sleep(0.05)
            progress_bar.progress((i + 1) / 40)

        # AutoTS real
        with st.spinner("⏳ Ejecutando AutoTS (puede tardar unos minutos)..."):
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

        progress_bar.empty()
        st.success("✅ Modelos entrenados y predicción generada")

        # Resultados
        st.subheader("📈 Predicción")
        st.line_chart(forecast_df)

        fig, ax = plt.subplots()
        df.iloc[:, 0].plot(ax=ax, label="Histórico", color="blue")
        forecast_df.iloc[:, 0].plot(ax=ax, label="Predicción", color="orange", linestyle="--")
        plt.legend()
        st.pyplot(fig)

        # Mejor modelo
        st.subheader("🏆 Mejor modelo")
        st.markdown(f"**Modelo:** `{model.best_model_name}`")
        st.markdown("**Parámetros:**")
        st.json(model.best_model_params)

        # Métricas
        st.subheader("📊 Top 5 modelos y puntuaciones")
        results_df = model.results().sort_values("Score", ascending=False)
        st.dataframe(results_df[["Model", "TransformationParameters", "Score"]].head())

        # Número de modelos exitosos
        st.markdown(f"📌 Modelos evaluados exitosamente: **{len(results_df)}**")
        
        # AutoTS no expone errores de modelos descartados directamente
        with st.expander("ℹ️ Información"):
        st.markdown("ℹ️ Algunos modelos fueron descartados automáticamente durante la evaluación. "
                "AutoTS continúa con los modelos viables y elige el mejor sin necesidad de intervención."
                   )


else:
    st.warning("👈 Sube primero un archivo CSV con índice de fecha y al menos una columna de valores.")

