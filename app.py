import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv("Genz_Spending.csv")

# Prepare AI model
X_cluster = data[["Eating Out (USD)", "Online Shopping (USD)", "Savings (USD)"]]
kmeans = KMeans(n_clusters=4)
data["Personality"] = kmeans.fit_predict(X_cluster)

# Train prediction model
X = data.drop("Personality", axis=1)
y = data["Personality"]

model = RandomForestClassifier()
model.fit(X, y)

# Personality mapping
personality_map = {
    0: "Saver",
    1: "Spender",
    2: "Investor",
    3: "Impulsive"
}


st.title("💰 Gen-Z Spending Behavior Analyzer")
id = st.number_input("ID",0)
age = st.number_input("Age", 18, 30)
income = st.number_input("Income (USD)", 0)
rent = st.number_input("Rent (USD)", 0)
groceries = st.number_input("Groceries (USD)", 0)
eating_out = st.number_input("Eating Out (USD)", 0)
entertainment = st.number_input("Entertainment (USD)", 0)
subscription = st.number_input("Subscription Services (USD)", 0)
education = st.number_input("Education (USD)", 0)
online_shopping = st.number_input("Online Shopping (USD)", 0)
savings = st.number_input("Savings (USD)", 0)
investments = st.number_input("Investments (USD)", 0)
travel = st.number_input("Travel (USD)", 0)
fitness = st.number_input("Fitness (USD)", 0)
miscellaneous = st.number_input("Miscellaneous (USD)", 0)

if st.button("Analyze Personality"):
    user_data = [[
        id,age, income, rent, groceries,eating_out,
        entertainment, subscription, education, online_shopping,savings,investments,travel,fitness,miscellaneous
    ]]
    
    prediction = model.predict(user_data)
    personality = personality_map[prediction[0]]
    
    st.subheader("🧠 Money Personality:")
    st.success(personality_map)
    
if personality_map == "Saver":
    st.info("Great! You manage money well. Consider investments.")
elif personality_map == "Spender":
    st.warning("Try budgeting and reducing unnecessary expenses.")
elif personality_map == "Investor":
    st.success("Excellent! Diversify your investments.")
else:
    st.error("Track expenses and control impulse buying.")

st.bar_chart(data[["eating_out","online_shopping","savings"]])


