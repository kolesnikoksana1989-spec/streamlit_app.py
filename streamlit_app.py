import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Прогноз продаж", layout="centered")

# стиль
st.markdown(
    "<style>body {background-color: white;} .stMetric {color: red;} "
    ".stSlider label, .stSelectbox label {color: #c40000;} .stButton>button {background-color:#c40000; color:white;} "
    "</style>", unsafe_allow_html=True)

st.title("📊 Прогноз продаж по бюджету и сезону")

df = pd.read_csv("data/ads_sales.csv", encoding="utf-8-sig")
season_map = {"зима": 0, "весна": 1, "лето": 2, "осень": 3}
df["Сезон_код"] = df["Сезон"].map(season_map)

X = df[["Бюджет_рекламы", "Сезон_код"]]
y = df["Продажи"]
model = LinearRegression().fit(X, y)

# боковое меню
st.sidebar.header("⚙️ Настройки прогноза")
budget = st.sidebar.slider("Бюджет рекламы (₽)", 5000, 40000, 20000, step=1000)
season = st.sidebar.selectbox("Сезон", list(season_map.keys()))
weeks = st.sidebar.slider("Прогноз на (недель)", 1, 8, 4)

season_code = season_map[season]
base_pred = model.predict([[budget, season_code]])[0]

st.subheader("💰 Прогноз продаж")
st.metric("Ожидаемый объём продаж", f"{base_pred:,.0f} ₽")

# прогноз на выбранное количество недель
future_weeks = np.arange(weeks)
trend = base_pred + (future_weeks * (budget / 5000) * 100)
future_dates = pd.date_range("2025-03-10", periods=weeks, freq="W-MON")
forecast_df = pd.DataFrame({"Дата": future_dates, "Прогноз_продаж": trend})

# график
fig, ax = plt.subplots(figsize=(7,4))
for s, g in df.groupby("Сезон"):
    ax.scatter(g["Бюджет_рекламы"], g["Продажи"], label=s)
x_range = np.linspace(df["Бюджет_рекламы"].min(), df["Бюджет_рекламы"].max(), 50)
y_pred = model.predict(np.column_stack([x_range, np.full_like(x_range, season_code)]))
ax.plot(x_range, y_pred, color="red", label=f"Прогноз ({season})")
ax.set_xlabel("Бюджет рекламы, ₽")
ax.set_ylabel("Продажи, ₽")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.subheader("📅 Прогноз продаж на ближайшие недели")
st.line_chart(forecast_df.set_index("Дата"))

st.caption("Модель учитывает влияние бюджета и сезона. Демонстрационный прогноз.")
