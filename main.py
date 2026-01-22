import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 1️⃣ App Title & Description
st.set_page_config(page_title="Smart Loan Approval System", page_icon="💰", layout="centered")
st.title("🏦 Smart Loan Approval System")
st.markdown("> *This system uses Support Vector Machines to predict loan approval.*")

# --- Data Loading and Preprocessing ---
@st.cache_resource
def load_and_train_model(kernel_type):
    # Dataset URL
    url = "https://raw.githubusercontent.com/Paliking/ML_examples/master/LoanPrediction/train_u6lujuX_CVtuZ9i.csv"
    df = pd.read_csv(url)

    # Features used
    features = ['ApplicantIncome', 'LoanAmount', 'Credit_History',
                'Education', 'Property_Area', 'Self_Employed']
    X = df[features].copy()
    y = df['Loan_Status'].map({'Y': 1, 'N': 0})

    # Encode categorical columns
    X['Education'] = X['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    X['Property_Area'] = X['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
    X['Self_Employed'] = X['Self_Employed'].map({'No': 0, 'Yes': 1})

    # Handle missing numeric values
    num_cols = ['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Self_Employed']
    imputer_num = SimpleImputer(strategy='median')
    X[num_cols] = imputer_num.fit_transform(X[num_cols])

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train SVM
    kernel_map = {"Linear SVM": "linear", "Polynomial SVM": "poly", "RBF SVM": "rbf"}
    model = SVC(kernel=kernel_map[kernel_type], probability=True, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler


# -------------------- UI INPUT BLOCKS (CENTERED) --------------------

st.subheader("📝 Applicant Information")
st.write("Fill in the details below:")

# Create centered layout
col_empty_left, col_main, col_empty_right = st.columns([1, 2, 1])

with col_main:
    # Block 1 – Income & Loan Details
    with st.container():
        st.markdown("### 💵 Financial Details")
        app_income = st.number_input("Applicant Income", min_value=0, value=5000, step=100)
        loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=120, step=10)
        credit_history = st.radio("Credit History", ["Yes", "No"], horizontal=True)

    st.divider()

    # Block 2 – Profile & Area
    with st.container():
        st.markdown("### 👤 Applicant Profile")
        employment = st.selectbox("Employment Status", ["Employed", "Self-Employed"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])


# -------------------- MODEL SELECTION --------------------

st.divider()
st.subheader("⚙️ Model Configuration")

kernel_choice = st.radio(
    "Select SVM Kernel:",
    ("Linear SVM", "Polynomial SVM", "RBF SVM"),
    horizontal=True,
)

model, scaler = load_and_train_model(kernel_choice)


# -------------------- PREDICTION BUTTON --------------------

st.divider()

center_btn_col1, center_btn_col2, center_btn_col3 = st.columns([1, 2, 1])
with center_btn_col2:
    predict_btn = st.button("Check Loan Eligibility", use_container_width=True)


# -------------------- PREDICTION OUTPUT --------------------

if predict_btn:
    ch_val = 1.0 if credit_history == "Yes" else 0.0
    edu_val = 1 if education == "Graduate" else 0
    prop_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    self_emp = 1 if employment == "Self-Employed" else 0
    
    user_data = np.array([[app_income, loan_amount, ch_val, edu_val,
                           prop_map[property_area], self_emp]])
    user_data_scaled = scaler.transform(user_data)
    
    prediction = model.predict(user_data_scaled)[0]
    confidence = model.predict_proba(user_data_scaled)[0]

    st.divider()

    if prediction == 1:
        st.success("### ✅ Loan Approved")
        st.balloons()
        status_text = "likely"
    else:
        st.error("### ❌ Loan Rejected")
        status_text = "unlikely"

    col1, col2 = st.columns(2)
    col1.metric("Confidence Score", f"{round(max(confidence) * 100, 2)}%")
    col2.metric("Kernel Applied", kernel_choice)

    st.subheader("💡 Business Explanation")
    st.info(f"""
    **Analysis:** Based on the applicant's credit history and income pattern, 
    the system determines the applicant is **{status_text}** to repay the loan. 
    
    *The {kernel_choice} kernel created the best decision boundary for this prediction.*
    """)


st.caption("Database Source: train.csv (Loan Prediction Problem Dataset)")
