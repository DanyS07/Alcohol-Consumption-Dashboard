import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# -----------------------------
# PAGE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="Alcohol Consumption Dashboard",
    page_icon="🍺",
    layout="wide"
)

# -----------------------------
# LOAD FILES
# -----------------------------

model = pickle.load(open("model.pkl","rb"))
country_encoder = pickle.load(open("country_encoder.pkl","rb"))
feature_columns = pickle.load(open("feature_columns.pkl","rb"))

df = pd.read_csv("beer-servings.csv", index_col=0)

# -----------------------------
# CONTINENT → COUNTRY MAP
# -----------------------------

continent_country_map = (
    df.groupby("continent")["country"]
    .apply(list)
    .to_dict()
)

# -----------------------------
# PAGE HEADER
# -----------------------------

st.title("🍺 Global Alcohol Consumption Analytics")

st.markdown(
"""
Interactive dashboard to explore global beverage patterns and  
predict **total litres of pure alcohol consumption**.
"""
)

st.divider()

# -----------------------------
# ANALYTICS SECTION
# -----------------------------

col1,col2,col3 = st.columns(3)

beer_total = df["beer_servings"].sum()
spirit_total = df["spirit_servings"].sum()
wine_total = df["wine_servings"].sum()

with col1:

    fig = px.pie(
        values=[beer_total,spirit_total,wine_total],
        names=["Beer","Spirit","Wine"],
        title="Beverage Share"
    )

    st.plotly_chart(fig,use_container_width=True)

with col2:

    fig = px.bar(
        df.groupby("continent")["total_litres_of_pure_alcohol"]
        .mean()
        .reset_index(),
        x="continent",
        y="total_litres_of_pure_alcohol",
        color="continent",
        title="Alcohol Consumption by Continent"
    )

    st.plotly_chart(fig,use_container_width=True)

with col3:

    fig = px.histogram(
        df,
        x="beer_servings",
        nbins=25,
        title="Beer Consumption Distribution"
    )

    st.plotly_chart(fig,use_container_width=True)

st.divider()

# -----------------------------
# LOCATION SELECTION
# -----------------------------

st.markdown("## 🌍 Location")

col1,col2 = st.columns(2)

with col1:

    continent = st.selectbox(
        "Select Continent",
        sorted(continent_country_map.keys())
    )

with col2:

    country = st.selectbox(
        "Select Country",
        continent_country_map[continent]
    )

st.caption(
f"{len(continent_country_map[continent])} countries available in {continent}"
)

# -----------------------------
# SERVING INPUT
# -----------------------------

st.markdown("## 🍷 Beverage Servings")

left,right = st.columns(2)

with left:

    beer = st.slider("Beer Servings",0,400,50)

    spirit = st.slider("Spirit Servings",0,400,50)

with right:

    wine = st.slider("Wine Servings",0,400,50)

st.divider()

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Predict Alcohol Consumption",use_container_width=True):

    country_encoded = country_encoder.transform([country])[0]

    data = pd.DataFrame({
        "country":[country_encoded],
        "beer_servings":[beer],
        "spirit_servings":[spirit],
        "wine_servings":[wine],
        "continent":[continent]
    })

    data = pd.get_dummies(data, columns=["continent"])

    # ensure all training columns exist
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0

    data = data[feature_columns]

    prediction = model.predict(data)[0]

    st.success(
        f"Estimated Alcohol Consumption: **{prediction:.2f} litres per person**"
    )

st.divider()