import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from autots import AutoTS
import time

st.set_page_config(page_title="Predicción de Series Temporales", layout="centered")
st.title("⏳ Predicción automática de series temporales")

st.sidebar.header("⚙️ Configuración del modelo")
forecast_length = st.sidebar.slider("Horizonte de predicción (días)", 7, 90, 30)
max_generations = st.sidebar.slider("Generaciones (iteraciones)", 1, 10, 5)

uploaded_file = st.file_uploader("📂 Sube tu archivo CSV con la serie temporal", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep=',', quotechar='"')
        st.success("✅ Archivo cargado correctamente")
        st.write("Vista previa de los datos:")
        st.dataframe(df.head())

        if df.shape[1] < 2:
            st.error("❌ El archivo debe contener al menos dos columnas: fecha y valor numérico.")
            st.stop()

        date_column = df.columns[0]
        value_column = df.columns[1]

        df[date_column] = df[date_column].astype(str).str.replace('"', '').str.strip()

        parsed = False
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                df[date_column] = pd.to_datetime(df[date_column], format=fmt)
                parsed = True
                break
            except Exception:
                continue

        if not parsed:
            try:
                df[date_column] = pd.to_datetime(df[date_column], format='mixed')
                parsed = True
            except Exception as e:
                st.error(f"❌ No se pudo convertir la columna de fechas automáticamente. Error: {e}")
                st.stop()

        df.set_index(date_column, inplace=True)

        if not isinstance(df.index, pd.DatetimeIndex):
            st.error("❌ La columna de fecha no pudo ser convertida en un índice temporal válido.")
            st.stop()

        if len(df) < forecast_length + 10:
            st.error("❌ No hay suficientes datos para entrenar el modelo con el horizonte seleccionado.")
            st.stop()

        if st.button("🚀 Ejecutar predicción"):
            start_time = time.time()
            with st.spinner("Entrenando modelos, por favor espera..."):
                model = AutoTS(
                    forecast_length=forecast_length,
                    frequency='infer',
                    ensemble='simple',
                    max_generations=max_generations,
                    num_validations=2,
                    validation_method="backwards"
                )
                model = model.fit(df)
                prediction = model.predict()
                forecast_df = prediction.forecast
            elapsed_time = time.time() - start_time
            st.success(f"✅ Predicción generada en {elapsed_time:.2f} segundos")

            # Mostrar top modelos
            st.subheader("📊 Top 5 modelos y puntuaciones")
            model_results = model.results()
            top_models = model_results.sort_values("Score", ascending=True).head(5)
            top_models_display = top_models[["Model", "Score"]].copy()
            st.table(top_models_display.reset_index(drop=True))

            # Mostrar parámetros técnicos en un expander
            with st.expander("⚙️ Parámetros técnicos de modelos"):
                st.dataframe(top_models[["Model", "TransformationParameters"]])

            # Visualización de predicción con IC
            st.subheader("📈 Predicción vs Histórico")
            plt.figure(figsize=(10, 5))
            plt.plot(df[value_column].iloc[-60:], label="Histórico", color="blue")
            plt.plot(forecast_df[value_column], label="Predicción", linestyle="--", color="orange")

            # Si hay IC disponibles:
            if prediction.lower_forecast is not None and prediction.upper_forecast is not None:
                plt.fill_between(
                    forecast_df.index,
                    prediction.lower_forecast[value_column],
                    prediction.upper_forecast[value_column],
                    color='orange', alpha=0.2, label='Intervalo de confianza'
                )

            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
