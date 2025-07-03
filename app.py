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
    df = pd.read_csv(uploaded_file, sep=',', quotechar='"')

    # Intentar convertir primera columna a índice de fechas con múltiples formatos
    date_column = df.columns[0]
    parse_success = False
    date_formats = ["%Y-%m-%d", "%Y-%m", "%Y"]

    for fmt in date_formats:
        try:
            df[date_column] = df[date_column].astype(str).str.replace('"', '').str.strip()
            df[date_column] = pd.to_datetime(df[date_column], format=fmt)
            df.set_index(date_column, inplace=True)
            parse_success = True
            break
        except Exception:
            continue

    if not parse_success:
        try:
            df[date_column] = df[date_column].astype(str).str.replace('"', '').str.strip()
            df[date_column] = pd.to_datetime(df[date_column], errors='raise')
            df.set_index(date_column, inplace=True)
            parse_success = True
        except Exception as e:
            st.error(f"❌ No se pudo convertir la columna de fechas automáticamente. Error: {e}")
            st.stop()

    if not isinstance(df.index, pd.DatetimeIndex):
        st.error("❌ El índice del dataset no es una serie temporal válida. Asegúrate de que la primera columna tenga fechas.")
        st.stop()

    st.subheader("📄 Vista previa de los datos")
    st.dataframe(df.head())

    # Validación inicial
    total_obs = df.shape[0]
    min_obs_required = 20
    max_safe_horizon = max(1, total_obs // 3)

    if total_obs < min_obs_required:
        st.error(f"❌ El dataset tiene solo {total_obs} observaciones. Se requieren al menos {min_obs_required} para realizar predicciones confiables.")
        st.stop()

    # Configuración
    st.sidebar.header("⚙️ Parámetros")
    forecast_length = st.sidebar.slider("Horizonte de predicción", 1, max_safe_horizon, min(30, max_safe_horizon))
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

        # Tiempo de entrenamiento
        start = time.perf_counter()

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
                    num_validations=1,
                    verbose=True
                )
                model = model.fit(df)
                prediction = model.predict()
                forecast_df = prediction.forecast
                upper = prediction.upper_forecast
                lower = prediction.lower_forecast

        except ValueError as e:
            if "forecast_length is too large for training data" in str(e):
                st.error("❌ Error: El horizonte de predicción es demasiado largo para los datos disponibles.\n"
                         "Prueba reduciendo el `forecast_length`, agregando más datos, o ajustando la configuración de validación.")
            else:
                st.exception(e)
            st.stop()

        duration = time.perf_counter() - start
        st.success(f"✅ Modelos entrenados en {duration:.2f} segundos.")

        st.subheader("📈 Predicción con intervalo de confianza")
        fig, ax = plt.subplots()
        df.iloc[:, 0].plot(ax=ax, label="Histórico", color="blue")
        forecast_df.iloc[:, 0].plot(ax=ax, label="Predicción", color="orange", linestyle="--")
        if upper is not None and lower is not None:
            ax.fill_between(upper.index, lower.iloc[:, 0], upper.iloc[:, 0], color='orange', alpha=0.2, label='Intervalo de confianza')
        plt.legend()
        st.pyplot(fig)

        st.subheader("🏆 Mejor modelo")
        friendly_names = {
            "SeasonalNaive": "Modelo estacional básico",
            "GLS": "Regresión lineal generalizada",
            "BasicLinearModel": "Modelo lineal básico",
        }
        modelo_nombre = friendly_names.get(model.best_model_name, model.best_model_name)
        st.markdown(f"**Modelo seleccionado:** `{modelo_nombre}`")

        with st.expander("🔍 Parámetros técnicos del modelo"):
            st.json(model.best_model_params)

        st.subheader("📊 Mejores modelos")
        results_df = model.results().sort_values("Score", ascending=False)
        simplified = results_df[["Model", "Score"]].head()
        simplified["Model"] = simplified["Model"].apply(lambda x: friendly_names.get(x, x))
        st.table(simplified.reset_index(drop=True))

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


