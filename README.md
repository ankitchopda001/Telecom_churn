# Telecom Churn Prediction

## 📊 Project Overview

This project aims to predict customer churn in the telecom industry using machine learning techniques. By analyzing customer behavior patterns, demographic information, and usage statistics, the model helps identify customers who are likely to discontinue their services, enabling proactive retention strategies.

## 🎯 Business Problem

Customer churn is a critical challenge in the telecom industry. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project develops a predictive model to identify customers at risk of churning, allowing telecom companies to:
- Implement targeted retention campaigns
- Reduce customer acquisition costs
- Improve customer lifetime value
- Enhance business revenue

## 📁 Dataset Information

The dataset contains customer information from a telecom company with the following features:

### Features:
- `customer_id`: Unique customer identifier
- `telecom_partner`: Telecom service provider (Reliance Jio, Vodafone, Airtel, BSNL)
- `gender`: Customer gender
- `age`: Customer age
- `state`: State of residence
- `city`: City of residence
- `pincode`: Postal code
- `date_of_registration`: Registration date
- `num_dependents`: Number of dependents
- `estimated_salary`: Estimated annual salary
- `calls_made`: Number of calls made
- `sms_sent`: Number of SMS sent
- `data_used`: Data usage in MB
- `churn`: Target variable (0 = No churn, 1 = Churn)

## 🛠️ Technologies Used

- **Python 3.12**
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms and preprocessing
- **XGBoost** - Gradient boosting implementation
- **Imbalanced-learn** - Handling class imbalance
- **Matplotlib & Seaborn** - Data visualization
- **Joblib** - Model serialization
- **FastAPI** - API development for model deployment
- **PyMySQL** - Database connectivity

## 📈 Exploratory Data Analysis (EDA)

Key insights from the EDA:

1. **Class Distribution**: Initial dataset showed class imbalance with approximately 80% non-churn and 20% churn customers
2. **Feature Analysis**:
   - Age distribution shows customers between 18-74 years
   - Usage patterns vary significantly among customers
   - State-wise distribution: Uttarakhand, Karnataka, Maharashtra have highest customer count
3. **Correlation Analysis**: Weak correlations between features, indicating churn is influenced by multiple factors

## 🔧 Data Preprocessing

### Steps Performed:

1. **Data Sampling**: Reduced dataset to 50,000 records for computational efficiency
2. **Feature Removal**: Dropped `customer_id` (non-predictive identifier)
3. **Categorical Encoding**: Used Label Encoding for categorical features
4. **Feature-Target Split**: Separated features (X) and target (y)
5. **Handling Class Imbalance**: Applied upsampling to balance churn and non-churn classes

### Feature Categories:

**Numerical Features**:
- age
- pincode
- num_dependents
- estimated_salary
- calls_made
- sms_sent
- data_used

**Categorical Features**:
- telecom_partner
- gender
- state
- city

## 🤖 Models Implemented

### 1. **Random Forest Classifier**
- **Accuracy**: 97.67%
- **Precision**: 0.96 (class 0), 0.99 (class 1)
- **Recall**: 1.00 (class 0), 0.96 (class 1)
- **F1-Score**: 0.98 (both classes)
- **ROC-AUC**: 0.9786

### 2. **XGBoost Classifier**
- **Accuracy**: 71.58%

### 3. **Logistic Regression**
- **Accuracy**: 49.89%

## 📊 Model Evaluation

### Confusion Matrix Analysis
- **True Positives**: High number of correctly predicted churn cases
- **True Negatives**: High number of correctly predicted non-churn cases
- **False Positives**: Low (few loyal customers incorrectly marked as churn)
- **False Negatives**: Low (few churn customers missed)

### ROC Curve
- AUC Score: 0.9786 indicates excellent model performance
- High True Positive Rate with low False Positive Rate

## 💻 Model Deployment

The best-performing model (Random Forest) was saved using joblib and deployed with FastAPI:

### API Endpoint: `/predict`
**Input Format**:
```json
{
  "telecom_partner": 2,
  "gender": 1,
  "age": 35,
  "state": 10,
  "city": 3,
  "pincode": 560001,
  "num_dependents": 2,
  "estimated_salary": 85000,
  "calls_made": 50,
  "sms_sent": 25,
  "data_used": 5000
}
