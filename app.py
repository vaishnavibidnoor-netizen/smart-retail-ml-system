import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Smart Retail ML System", layout="wide")

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown("""
# 🛍 Smart Retail Sales Intelligence Dashboard
""")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📌 Navigation")

dataset_size = st.sidebar.slider("Dataset Size", 1000, 5000, 2000, step=500)

page = st.sidebar.radio("Go to", [
    "🏠 Project Description",
    "🏠 Executive Dashboard",
    "📊 Data Insights",
    "📈 Linear Regression - Sales Forecast",
    "🌳 Decision Tree - Purchase Decision",
    "🤝 KNN - Loyalty Prediction",
    "📌 K-Means - Customer Segmentation"
])

# --------------------------------------------------
# SYNTHETIC DATA
# --------------------------------------------------

@st.cache_data
def generate_data(n=2000):
    np.random.seed(42)

    data = pd.DataFrame({
        "Age": np.random.randint(18, 65, n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Annual Income": np.random.randint(20000, 120000, n),
        "Purchase Frequency": np.random.randint(1, 20, n),
        "Discount Offered": np.random.randint(0, 50, n),
        "Marketing Spend": np.random.randint(1000, 20000, n),
        "Seasonal Demand Index": np.random.uniform(0.5, 1.5, n),
        "Product Category": np.random.choice(
            ["Electronics", "Clothing", "Grocery", "Furniture"], n
        ),
        "Store Location": np.random.choice(
            ["Urban", "Semi-Urban", "Rural"], n
        )
    })

    category_weight = {"Electronics": 1.3,"Clothing": 1.1,"Grocery": 0.9,"Furniture": 1.2}
    location_weight = {"Urban": 1.2,"Semi-Urban": 1.0,"Rural": 0.8}

    data["Monthly Sales"] = (
        data["Annual Income"] * 0.03 +
        data["Purchase Frequency"] * 300 +
        data["Marketing Spend"] * 0.2 +
        data["Discount Offered"] * 40 +
        data["Seasonal Demand Index"] * 1000 +
        data["Product Category"].map(category_weight) * 500 +
        data["Store Location"].map(location_weight) * 400 +
        np.random.normal(0, 1000, n)
    )

    data["Purchase Decision"] = np.where(data["Purchase Frequency"] > 10, 1, 0)

    data["Loyal Customer"] = np.where(
        (data["Annual Income"] > 70000) &
        (data["Purchase Frequency"] > 12), 1, 0
    )

    return data


data = generate_data(dataset_size)

# Encode categorical
le1, le2, le3 = LabelEncoder(), LabelEncoder(), LabelEncoder()
data["Gender"] = le1.fit_transform(data["Gender"])
data["Product Category"] = le2.fit_transform(data["Product Category"])
data["Store Location"] = le3.fit_transform(data["Store Location"])

# --------------------------------------------------
# PROJECT DESCRIPTION PAGE
# --------------------------------------------------

if page == "🏠 Project Description":

    st.subheader("🛍 Smart Retail Sales Intelligence System")

    st.markdown("### 📌 Problem Statement")
    st.write("""
Retail businesses struggle with forecasting sales, identifying loyal customers,
and understanding purchasing behavior. Traditional systems lack predictive intelligence.
    """)

    st.markdown("### 🎯 Project Objectives")
    st.write("""
• Predict Monthly Sales using Linear Regression  
• Classify Purchase Decisions using Decision Tree  
• Identify Loyal Customers using KNN  
• Segment Customers using K-Means Clustering  
• Provide interactive data visualization dashboard  
    """)

    st.markdown("### 🧠 Technologies Used")
    st.write("""
Python, Streamlit, Scikit-Learn, Pandas, NumPy,
Matplotlib, Seaborn
    """)

    st.markdown("### 💡 Business Impact")
    st.write("""
• Improves revenue forecasting  
• Enhances customer retention  
• Optimizes marketing strategies  
• Supports data-driven decision making  
    """)

    st.markdown("### 👩‍💻 Team Details")
    st.write("""
Team Name: InsightX  
Team Member: Vaishnavi  
Department: AIML  
Event: Internal Data Science Hackathon 2026  
    """)

    st.success("🚀 This project satisfies hackathon requirements.")

# --------------------------------------------------
# EXECUTIVE DASHBOARD
# --------------------------------------------------

elif page == "🏠 Executive Dashboard":

    st.subheader("📌 Business Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Total Customers", len(data))
    col2.metric("💰 Avg Monthly Sales", f"₹ {int(data['Monthly Sales'].mean())}")
    col3.metric("⭐ Loyal Customers", int(data["Loyal Customer"].sum()))

    st.markdown("---")

    col4, col5 = st.columns(2)

    with col4:
        fig, ax = plt.subplots()
        sns.histplot(data["Monthly Sales"], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

    with col5:
        fig, ax = plt.subplots()
        sns.countplot(x=data["Product Category"], ax=ax)
        st.pyplot(fig)

# --------------------------------------------------
# DATA INSIGHTS
# --------------------------------------------------

elif page == "📊 Data Insights":

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(data.corr(), cmap="coolwarm", annot=False)
    st.pyplot(fig)



# --------------------------------------------------
# LINEAR REGRESSION
# --------------------------------------------------

elif page == "📈 Linear Regression - Sales Forecast":

    X = data.drop(["Monthly Sales","Purchase Decision","Loyal Customer"], axis=1)
    y = data["Monthly Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, predictions)

    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2:.2f}")
    col2.metric("MSE", f"{int(mse)}")

    # 📊 GRAPH
    fig, ax = plt.subplots(figsize=(8,6))

    # Scatter plot
    ax.scatter(y_test, predictions, alpha=0.6)

    # Perfect prediction reference line
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red",
        linewidth=2
    )

    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Actual vs Predicted Sales")

    st.pyplot(fig)

# --------------------------------------------------
# DECISION TREE
# --------------------------------------------------

elif page == "🌳 Decision Tree - Purchase Decision":

    X = data.drop(["Monthly Sales","Purchase Decision","Loyal Customer"], axis=1)
    y = data["Purchase Decision"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train,y_train)

    predictions = clf.predict(X_test)

    st.metric("Accuracy",f"{accuracy_score(y_test,predictions)*100:.2f}%")

    # 📊 Feature Importance Graph
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": clf.feature_importances_
    }).sort_values(by="Importance")

    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(importance["Feature"], importance["Importance"])
    ax.set_title("Feature Importance - Decision Tree")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)


# --------------------------------------------------
# KNN
# --------------------------------------------------

elif page == "🤝 KNN - Loyalty Prediction":

    X = data.drop(["Monthly Sales","Purchase Decision","Loyal Customer"], axis=1)
    y = data["Loyal Customer"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train,y_train)

    predictions = knn.predict(X_test)

    st.metric("Accuracy",f"{accuracy_score(y_test,predictions)*100:.2f}%")

    # 📊 Confusion Matrix Graph
    cm = confusion_matrix(y_test,predictions)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    ax.set_title("Confusion Matrix - KNN")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# --------------------------------------------------
# K-MEANS
# --------------------------------------------------

elif page == "📌 K-Means - Customer Segmentation":

    scaler = StandardScaler()
    segment_data = data[["Annual Income","Purchase Frequency","Marketing Spend"]]
    scaled = scaler.fit_transform(segment_data)

    kmeans = KMeans(n_clusters=3,random_state=42)
    data["Segment"] = kmeans.fit_predict(scaled)

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=data,
        x="Annual Income",
        y="Purchase Frequency",
        hue="Segment"
    )
    st.pyplot(fig)

    st.write("### Segment Summary")
    st.write(data.groupby("Segment")[["Annual Income","Purchase Frequency"]].mean())

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
---
Made with  by Vaishnavi | AI & ML Project
""")