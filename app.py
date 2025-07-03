import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from autots import AutoTS
import time

st.set_page_config(page_title="AutoTS WebApp", layout="centered")
st.title("\ud83d\udcc8 Predicci\u00f3n autom\u00e1tica de series temporales con AutoTS")

# Subida de archivo
uploaded_file = st.file_uploader("\ud83d\udcce Sube un archivo CSV con la serie temporal", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
    st.subheader("\ud83d\udcc4 Vista previa de los datos")
    st.dataframe(df.head())

    # Configuraci\u00f3n
    st.sidebar.header("\u2699\ufe0f Par\u00e1metros")
    max_horizon = max(7, df.shape[0] // 3)
    forecast_length = st.sidebar.slider("Horizonte de predicci\u00f3n", 1, max_horizon, min(30, max_horizon))
    ensemble_mode = st.sidebar.selectbox("Tipo de Ensemble", ["simple", "best", "auto"])
    model_list_option = st.sidebar.selectbox("Lista de modelos", ["superfast", "fast", "default", "all"])
    max_generations = st.sidebar.slider("Generaciones (iteraciones)", 1, 10, 5)

    if st.button("\ud83d\ude80 Ejecutar predicci\u00f3n"):
        estimated_models = {
            "superfast": 30,
            "fast": 60,
            "default": 100,
            "all": 150
        }
        total_models = estimated_models.get(model_list_option, 50) * max_generations

        st.info(f"\ud83e\uddf6 Estimando alrededor de {total_models} modelos por evaluar...")

        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(40):
            time.sleep(0.05)
            progress_bar.progress((i + 1) / 40)

        # Validaci\u00f3n defensiva
        min_required_obs = int(forecast_length * 2.5)
        if df.shape[0] < min_required_obs:
            st.error(f"\u274c Tu serie solo tiene {df.shape[0]} observaciones, pero se requieren al menos {min_required_obs} "
                     f"para un horizonte de predicci\u00f3n de {forecast_length} d\u00edas. "
                     "Reduce el horizonte o agrega m\u00e1s datos.")
        else:
            with st.spinner("\u23f3 Ejecutando AutoTS (puede tardar unos minutos)..."):
                model = AutoTS(
                    forecast_length=forecast_length,
                    frequency='infer',
                    ensemble=ensemble_mode,
                    model_list=model_list_option,
                    transformer_list="fast",
                    max_generations=max_generations,
                    drop_most_recent=1,
                    validation_method="simple"
                )
                model = model.fit(df)
                prediction = model.predict()
                forecast_df = prediction.forecast

            progress_bar.empty()
            st.success("\u2705 Modelos entrenados y predicci\u00f3n generada")

            st.subheader("\ud83d\udcc8 Predicci\u00f3n")
            st.line_chart(forecast_df)

            fig, ax = plt.subplots()
            df.iloc[:, 0].plot(ax=ax, label="Hist\u00f3rico", color="blue")
            forecast_df.iloc[:, 0].plot(ax=ax, label="Predicci\u00f3n", color="orange", linestyle="--")
            plt.legend()
            st.pyplot(fig)

            st.subheader("\ud83c\udfc6 Mejor modelo")
            st.markdown(f"**Modelo:** `{model.best_model_name}`")
            st.markdown("**Par\u00e1metros:**")
            st.json(model.best_model_params)

            st.subheader("\ud83d\udcca Top 5 modelos y puntuaciones")
            results_df = model.results().sort_values("Score", ascending=False)
            st.dataframe(results_df[["Model", "TransformationParameters", "Score"]].head())

            st.markdown(f"\ud83d\udccc Modelos evaluados exitosamente: **{len(results_df)}**")

            with st.expander("\u2139\ufe0f Informaci\u00f3n"):
                st.markdown(
                    "\u2139\ufe0f Algunos modelos fueron descartados autom\u00e1ticamente durante la evaluaci\u00f3n. "
                    "AutoTS contin\u00faa con los modelos viables y elige el mejor sin necesidad de intervenci\u00f3n."
                )

            # Exportar CSV
            csv = forecast_df.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="\ud83d\udcc5 Descargar predicci\u00f3n en CSV",
                data=csv,
                file_name="forecast_autots.csv",
                mime='text/csv'
            )

else:
    st.warning("\ud83d\udc48 Sube primero un archivo CSV con \u00edndice de fecha y al menos una columna de valores.")

