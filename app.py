import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from autots import AutoTS
import time

st.set_page_config(page_title="AutoTS WebApp", layout="centered")
st.title("📈 Predicción automática de series temporales con AutoTS")

# Subida de archivo
uploaded_file = st.file_uploader("📎 Sube un archivo CSV con la serie temporal", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
    st.subheader("📄 Vista previa de los datos")
    st.dataframe(df.head())

    # Configuración
    st.sidebar.header("⚙️ Parámetros")
    max_horizon = max(7, df.shape[0] // 3)
    forecast_length = st.sidebar.slider("Horizonte de predicción", 1, max_horizon, min(30, max_horizon))
    ensemble_mode = st.sidebar.selectbox("Tipo de Ensemble", ["simple", "best", "auto"])
    model_list_option = st.sidebar.selectbox("Lista de modelos", ["superfast", "fast", "default", "all"])
    max_generations = st.sidebar.slider("Generaciones (iteraciones)", 1, 10, 5)

    if st.button("🚀 Ejecutar predicción"):
        estimated_models = {
            "superfast": 30,
            "fast": 60,
            "default": 100,
            "all": 150
        }
        total_models = estimated_models.get(model_list_option, 50) * max_generations

        st.info(f"🧶 Estimando alrededor de {total_models} modelos por evaluar...")

        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(40):
            time.sleep(0.05)
            progress_bar.progress((i + 1) / 40)

        # Validación defensiva
        min_required_obs = int(forecast_length * 2.5)
        if df.shape[0] < min_required_obs:
            st.error(f"❌ Tu serie solo tiene {df.shape[0]} observaciones, pero se requieren al menos {min_required_obs} "
                     f"para un horizonte de predicción de {forecast_length} días. "
                     "Reduce el horizonte o agrega más datos.")
        else:
            try:
                with st.spinner("⏳ Ejecutando AutoTS (puede tardar unos minutos)..."):
                    model = AutoTS(
                        forecast_length=forecast_length,
                        frequency='infer',
                        ensemble=ensemble_mode,
                        model_list=model_list_option,
                        transformer_list="fast",
                        max_generations=max_generations,
                        drop_most_recent=1,
                        validation_method="backwards",
                        num_validations=1
                    )
                    model = model.fit(df)
                    prediction = model.predict()
                    forecast_df = prediction.forecast
            except ValueError as e:
                if "forecast_length is too large for training data" in str(e):
                    st.error("❌ Error: El horizonte de predicción es demasiado largo para los datos disponibles.\n"
                             "Prueba reduciendo el `forecast_length`, agregando más datos, o ajustando la configuración de validación.")
                else:
                    st.exception(e)
                progress_bar.empty()
                st.stop()

            progress_bar.empty()
            st.success("✅ Modelos entrenados y predicción generada")

            st.subheader("📈 Predicción")
            st.line_chart(forecast_df)

            fig, ax = plt.subplots()
            df.iloc[:, 0].plot(ax=ax, label="Histórico", color="blue")
            forecast_df.iloc[:, 0].plot(ax=ax, label="Predicción", color="orange", linestyle="--")
            plt.legend()
            st.pyplot(fig)

            st.subheader("🏆 Mejor modelo")
            st.markdown(f"**Modelo:** `{model.best_model_name}`")
            st.markdown("**Parámetros:**")
            st.json(model.best_model_params)

            st.subheader("📊 Top 5 modelos y puntuaciones")
            results_df = model.results().sort_values("Score", ascending=False)
            st.dataframe(results_df[["Model", "TransformationParameters", "Score"]].head())

            st.markdown(f"📌 Modelos evaluados exitosamente: **{len(results_df)}**")

            with st.expander("ℹ️ Información"):
                st.markdown(
                    "ℹ️ Algunos modelos fueron descartados automáticamente durante la evaluación. "
                    "AutoTS continúa con los modelos viables y elige el mejor sin necesidad de intervención."
                )

            # Exportar CSV con codificación compatible
            csv = forecast_df.to_csv(index=True).encode('utf-8-sig')
            st.download_button(
                label="Descargar predicción en CSV",
                data=csv,
                file_name="forecast_autots.csv",
                mime='text/csv'
            )

else:
    st.warning("👈 Sube primero un archivo CSV con índice de fecha y al menos una columna de valores.")

else:
    st.warning("👈 Sube primero un archivo CSV con índice de fecha y al menos una columna de valores.")
