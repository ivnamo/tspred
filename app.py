import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from autots import AutoTS
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="Predicción de Series Temporales", layout="wide")
st.title("📈 Predicción automática de series temporales")

# Sidebar - Parámetros del modelo
st.sidebar.header("⚙️ Parámetros del modelo")
forecast_length = st.sidebar.number_input("Horizonte de predicción", 1, 365, 30)
max_generations = st.sidebar.slider("Generaciones (iteraciones)", 1, 10, 5)

# Cargar archivo CSV
uploaded_file = st.file_uploader("📂 Sube un archivo CSV con una serie temporal", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Archivo cargado correctamente")

        # Mostrar vista previa
        st.subheader("👁 Vista previa de los datos")
        st.dataframe(df.head())

        # Detectar columna de fechas y valores
        date_col, value_col = df.columns[:2]
        df[date_col] = df[date_col].astype(str)

        # Convertir fechas con soporte flexible
        parsed = False
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                df[date_col] = pd.to_datetime(df[date_col], format=fmt)
                parsed = True
                break
            except Exception:
                continue
        if not parsed:
            try:
                df[date_col] = pd.to_datetime(df[date_col], format="mixed", dayfirst=True)
                parsed = True
            except Exception as e:
                st.error(f"❌ No se pudo convertir la columna de fechas automáticamente. Error: {e}")
                st.stop()

        df = df[[date_col, value_col]].dropna()
        df.columns = ["datetime", "value"]
        df = df.set_index("datetime")

        # Verificación previa de longitud vs. forecast
        if len(df) < forecast_length * 2:
            st.warning("⚠️ Demasiados pocos datos para el horizonte de predicción seleccionado. Reduce el horizonte o añade más datos.")
            st.stop()

        if st.button("🚀 Ejecutar predicción"):
            with st.spinner("Entrenando modelos..."):
                progress_bar = st.progress(0)
                model = AutoTS(
                    forecast_length=forecast_length,
                    frequency='infer',
                    ensemble='simple',
                    max_generations=max_generations,
                    num_validations=2,
                    min_allowed_train_percent=0.5,
                    model_list="fast_parallel",
                    validation_method="backwards",
                    verbose=1
                )
                model = model.fit(df)
                prediction = model.predict()
                forecast_df = prediction.forecast
                upper = prediction.upper_forecast
                lower = prediction.lower_forecast

            st.success("✅ Modelos entrenados y predicción generada")

            # Mostrar métricas
            st.subheader("🏅 Modelos top")
            leaderboard = model.results().sort_values(by="Score")
            st.dataframe(leaderboard[["Model", "Score", "SMAPE"]].head(5))

            # Gráfica de predicción
            st.subheader("🔮 Predicción vs histórico")
            fig, ax = plt.subplots(figsize=(12, 5))
            df[-forecast_length * 2:].plot(ax=ax, label="Histórico")
            forecast_df.plot(ax=ax, label="Predicción")
            if not upper.empty and not lower.empty:
                ax.fill_between(forecast_df.index, lower.iloc[:, 0], upper.iloc[:, 0], color='orange', alpha=0.2, label='Intervalo de confianza')
            ax.set_title("Predicción de la serie temporal")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Valor")
            ax.legend()
            st.pyplot(fig)

            # Mostrar errores si los hay
            try:
                errors_df = model.failure_reason()
                if not errors_df.empty:
                    with st.expander("⚠️ Errores de modelos descartados"):
                        st.dataframe(errors_df[["Model", "Error Message"]])
            except Exception:
                pass

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
