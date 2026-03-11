**Alcohol Consumption Analytics Dashboard**
A data-driven dashboard built using Python, Machine Learning, and Streamlit to analyze global alcohol consumption patterns and predict alcohol consumption based on beverage servings.
The application combines interactive infographics with a predictive model to provide insights into alcohol consumption trends across countries and continents.

**Features**
Interactive data visualizations
Global alcohol consumption insights
Machine learning prediction model
Continent → Country dynamic selection
Distribution analysis of beer consumption
Comparison of alcohol consumption across continents

**Machine Learning Model**

Two models were tested during training:
Linear Regression
Random Forest Regressor
The model with the higher R² score was selected automatically for deployment.

**Project Structure**
alcohol-consumption-dashboard
│
├── app.py
├── train_model.py
├── beer-servings.csv
├── model.pkl
├── feature_columns.pkl
├── country_encoder.pkl
└── README.md
