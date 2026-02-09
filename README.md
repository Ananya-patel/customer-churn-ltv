# 📊 Customer Churn & LTV Prediction System

##  Problem Statement
Customer churn is a major revenue risk for subscription-based businesses.  
This project predicts **which customers are likely to churn** and **how valuable they are**, enabling data-driven retention strategies.

---

##  Solution Overview
I built an end-to-end machine learning system that:
- Predicts customer churn probability
- Estimates customer lifetime value (LTV)
- Segments customers into actionable business groups
- Deploys predictions via a live web application

---

##  Dataset
**Telco Customer Churn Dataset**
- Customer demographics
- Subscription details
- Monthly charges & tenure
- Churn label

---

##  Key Steps
1. Exploratory Data Analysis (EDA)
2. Feature Engineering (tenure buckets, contract risk, spend behavior)
3. Churn Modeling (Logistic Regression → Random Forest → XGBoost)
4. LTV Modeling (Linear Regression → Gradient Boosting)
5. Customer Segmentation (Churn × LTV)
6. Streamlit Web App Deployment

---

## Models & Metrics
### Churn Model
- Final Model: **XGBoost**
- Metric: **ROC-AUC**
- Focus: High recall for churned customers

### LTV Model
- Final Model: **Gradient Boosting Regressor**
- Metrics: **R², MAE**

---

## Business Segments
| Segment | Action |
|------|------|
| High LTV + High Churn | 🚨 Retain immediately |
| High LTV + Low Churn | 💎 VIP customers |
| Low LTV + High Churn | ⚠️ Low-cost retention |
| Low LTV + Low Churn | 🙂 Stable |

---

## 🌐 Live App
👉 **Live Demo:** https://customer-churn-ltv.streamlit.app

---

## 🛠 Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Streamlit
- GitHub + Streamlit Cloud

---

##  Key Takeaway
This project demonstrates how machine learning can directly support **revenue retention and decision-making**, not just model accuracy.
