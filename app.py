import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc

# Page config
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

# Load artifacts
@st.cache_resource
def load_artifacts():
    with open('fraud_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('optimal_threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)
    with open('feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return model, scaler, threshold, feature_cols

model, scaler, optimal_threshold, feature_cols = load_artifacts()

# Sidebar
with st.sidebar:
    st.title("💳 Fraud Detector")
    st.success("✅ Model Loaded")
    st.info(f"Optimal Threshold: **{optimal_threshold:.2f}**")
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. Upload a CSV with columns: Time, Amount, V1-V28
    2. Click **Analyze**
    3. View fraud predictions & download results
    """)

# Main app
st.title("🔍 Credit Card Fraud Detection System")

uploaded_file = st.file_uploader("📁 Upload your transaction CSV file", type=['csv'])
use_demo = st.checkbox("Use demo data (1000 transactions)", value=False)

if use_demo and not uploaded_file:
    df = pd.read_csv('test_sample.csv')
    st.success("✅ Loaded demo data!")
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df):,} transactions!")
else:
    st.info("👆 Upload a CSV file or check 'Use demo data' to get started.")
    st.stop()

# Validate columns
missing_cols = [col for col in feature_cols if col not in df.columns]
if missing_cols:
    st.error(f"❌ Missing required columns: {missing_cols}")
    st.stop()

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Transactions", f"{len(df):,}")
with col2:
    if 'Class' in df.columns:
        fraud_count = df['Class'].sum()
        st.metric("Actual Frauds", f"{fraud_count:,}", f"{fraud_count/len(df)*100:.2f}%")
    else:
        st.metric("Actual Frauds", "N/A")
with col3:
    st.metric("Avg Transaction", f"${df['Amount'].mean():.2f}")

st.dataframe(df.head(10), use_container_width=True)

# Predictions
st.markdown("---")
st.subheader("2️⃣ Fraud Detection")

with st.spinner("Analyzing transactions... 🔍"):
    X = df[feature_cols].copy()
    X[['Time', 'Amount']] = scaler.transform(X[['Time', 'Amount']])
    fraud_proba = model.predict_proba(X)[:, 1]
    fraud_pred = (fraud_proba >= optimal_threshold).astype(int)
    
    results = df.copy()
    results['Fraud_Probability'] = fraud_proba
    results['Predicted_Fraud'] = fraud_pred
    results['Risk_Level'] = pd.cut(fraud_proba, bins=[0, 0.3, 0.7, 1.0], labels=['🟢 Low', '🟡 Medium', '🔴 High'])

# Results summary
st.markdown("---")
st.subheader("3️⃣ Results Summary")

flagged = results[results['Predicted_Fraud'] == 1]
total_flagged = len(flagged)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Flagged as Fraud", f"{total_flagged:,}", f"{total_flagged/len(results)*100:.2f}%")
with c2:
    st.metric("High Risk", f"{len(results[results['Risk_Level']=='🔴 High']):,}")
with c3:
    st.metric("Avg Fraud Probability", f"{fraud_proba.mean():.3f}")
with c4:
    if 'Class' in results.columns:
        true_positives = ((results['Predicted_Fraud']==1) & (results['Class']==1)).sum()
        precision = true_positives / total_flagged if total_flagged > 0 else 0
        st.metric("Precision on Flags", f"{precision:.2%}")
    else:
        st.metric("Precision on Flags", "N/A")

# Visualizations
st.markdown("---")
st.subheader("4️⃣ Visualizations")

tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🔥 Risk Heatmap", "📋 Flagged Transactions"])

with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(results[results['Predicted_Fraud']==0]['Fraud_Probability'], bins=50, alpha=0.6, label='Predicted Normal', color='green', density=True)
    if total_flagged > 0:
        axes[0].hist(results[results['Predicted_Fraud']==1]['Fraud_Probability'], bins=50, alpha=0.6, label='Predicted Fraud', color='red', density=True)
    axes[0].axvline(optimal_threshold, color='black', linestyle='--', label=f'Threshold={optimal_threshold:.2f}')
    axes[0].set_xlabel('Fraud Probability')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Fraud Probability Distribution')
    axes[0].legend()
    
    if total_flagged > 0:
        axes[1].boxplot([results[results['Predicted_Fraud']==0]['Amount'], results[results['Predicted_Fraud']==1]['Amount']], labels=['Normal', 'Fraud'])
        axes[1].set_ylabel('Amount ($)')
        axes[1].set_title('Transaction Amount by Prediction')
    
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    top_risk = results.nlargest(20, 'Fraud_Probability')[['Time', 'Amount', 'Fraud_Probability', 'Predicted_Fraud', 'Risk_Level']]
    st.dataframe(top_risk, use_container_width=True)

with tab3:
    if total_flagged > 0:
        flagged_display = flagged[['Time', 'Amount', 'Fraud_Probability', 'Risk_Level'] + [f'V{i}' for i in range(1, 6)]]
        st.dataframe(flagged_display.sort_values('Fraud_Probability', ascending=False), use_container_width=True)
    else:
        st.info("✅ No transactions flagged as fraud.")

# Ground truth comparison
if 'Class' in results.columns:
    st.markdown("---")
    st.subheader("5️⃣ Model Performance")
    
    y_true = results['Class']
    y_pred = results['Predicted_Fraud']
    cm = confusion_matrix(y_true, y_pred)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.text("Classification Report:")
        st.text(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))
    
    with col_b:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Normal', 'Predicted Fraud'],
                    yticklabels=['Actual Normal', 'Actual Fraud'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)
    
    precision, recall, _ = precision_recall_curve(y_true, fraud_proba)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f'PR-AUC = {pr_auc:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Download results
st.markdown("---")
st.subheader("6️⃣ Download Results")

csv = results.to_csv(index=False).encode('utf-8')
st.download_button(label="📥 Download Full Results as CSV", data=csv, file_name='fraud_predictions.csv', mime='text/csv')

st.caption("Built with Streamlit | Model trained on Kaggle with SMOTE & Random Forest")