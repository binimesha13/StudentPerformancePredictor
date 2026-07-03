import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Score Predictor + Goal Optimizer", page_icon="🎯")

@st.cache_data
def load_data():
    df = pd.read_csv("student_data.csv")
    df.columns = [c.strip() for c in df.columns]
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("student_data.csv not found. Place the dataset in this folder.")
    st.stop()

st.title("🎯 Student Score Predictor + Goal Optimizer")
st.write("Not just predicting scores — this tool tells you **exactly what to change** to hit your target score.")
st.caption(f"Dataset loaded: {df.shape[0]} students, {df.shape[1]} columns")

score_cols = ["math score", "reading score", "writing score"]
target_col = st.selectbox("Select which score you want to predict", score_cols, index=0)

df_clean = df.dropna().copy()

cat_cols = df_clean.select_dtypes(include='object').columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    encoders[col] = le

X = df_clean.drop(columns=[target_col])
y = df_clean[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=150, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

st.subheader("📈 Model Comparison")
col1, col2 = st.columns(2)
with col1:
    st.metric("Linear Regression R²", f"{r2_score(y_test, lr_preds):.3f}")
    st.metric("Linear Regression MAE", f"{mean_absolute_error(y_test, lr_preds):.2f}")
with col2:
    st.metric("Random Forest R²", f"{r2_score(y_test, rf_preds):.3f}")
    st.metric("Random Forest MAE", f"{mean_absolute_error(y_test, rf_preds):.2f}")

best_model = rf if r2_score(y_test, rf_preds) > r2_score(y_test, lr_preds) else lr
best_name = "Random Forest" if best_model is rf else "Linear Regression"
st.success(f"Best performing model selected automatically: **{best_name}**")

st.divider()
st.subheader("🔮 Predict a Student's Score")

input_vals = {}
for col in X.columns:
    if col in encoders:
        options = list(encoders[col].classes_)
        choice = st.selectbox(col, options)
        input_vals[col] = encoders[col].transform([choice])[0]
    else:
        min_v, max_v = float(df_clean[col].min()), float(df_clean[col].max())
        mean_v = float(df_clean[col].mean())
        input_vals[col] = st.slider(col, min_v, max_v, mean_v)

input_df = pd.DataFrame([input_vals])[X.columns]
input_scaled = scaler.transform(input_df)
current_prediction = float(best_model.predict(input_scaled)[0])
current_prediction = float(np.clip(current_prediction, 0, 100))

st.info(f"Predicted {target_col.title()}: **{current_prediction:.2f} / 100**")

st.divider()
st.subheader("🎯 Target Score Optimizer")
st.write("Tell me the dream score — I'll tell you which factor to improve, and by how much, to get there.")

target_score = st.slider("Target Score", 0.0, 100.0, min(100.0, current_prediction + 10))

if st.button("Find Path to Target"):
    gap = target_score - current_prediction
    if gap <= 0:
        st.success("This student is already predicted to meet or exceed this target! 🎉")
    else:
        if best_name == "Linear Regression":
            importances = dict(zip(X.columns, best_model.coef_))
        else:
            importances = dict(zip(X.columns, best_model.feature_importances_))

        numeric_cols = [c for c in X.columns if c not in encoders]
        sorted_feats = sorted(
            [(f, importances[f]) for f in numeric_cols if importances[f] > 0],
            key=lambda x: -x[1]
        )

        if sorted_feats:
            top_feature, impact = sorted_feats[0]
            std_dev = df_clean[top_feature].std()
            needed_increase = gap / (abs(impact) + 1e-6) * 0.1 * std_dev
            current_val = input_vals[top_feature]
            new_val = min(100, current_val + needed_increase)

            st.success(
                f"To gain **{gap:.1f} more points** in {target_col}, try increasing "
                f"**{top_feature}** from **{current_val:.1f}** to approximately **{new_val:.1f}**."
            )
            st.caption(
                "This is the numeric factor with the strongest positive impact on the "
                "predicted score, based on the trained model."
            )
        else:
            st.warning("No positive-impact numeric features found to compute a recommendation.")

st.divider()
st.subheader("📊 What Drives the Score?")

if best_name == "Linear Regression":
    coeffs = best_model.coef_
else:
    coeffs = best_model.feature_importances_

coeff_df = pd.DataFrame({"Feature": X.columns, "Impact": coeffs}).sort_values(by="Impact", ascending=True)

fig, ax = plt.subplots()
ax.barh(coeff_df["Feature"], coeff_df["Impact"], color="skyblue")
ax.set_xlabel("Impact on Predicted Score")
st.pyplot(fig)

st.caption("Built with Python, Scikit-learn (Linear Regression + Random Forest), and Streamlit.")