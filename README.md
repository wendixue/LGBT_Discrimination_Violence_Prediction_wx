# Unpacking Identity Vulnerability  
**Predicting Discrimination and Violence against LGBTIQ+ Individuals**

## Overview
This project investigates how **gender identity, economic status, and multiple minority identities** influence the likelihood of experiencing discrimination and violence among LGBTIQ+ individuals. Using machine learning models, the study identifies key predictors of vulnerability and cross-national variation in risk.

## Data
- **EU LGBTIQ Survey III (2023)**  
- N = 100,577 respondents  
- Variables: demographics, socioeconomic conditions, health, life satisfaction, identity openness, disclosure, and perceived safety.  
- Outcomes:  
  - **Discrimination** across employment, housing, healthcare, education, public spaces, and government services.  
  - **Violence** (frequency of physical or sexual attacks in the last 5 years).  

## Methods
- **Feature Engineering**: demographic, socioeconomic, and identity-related predictors.  
- **Models**:  
  - Discrimination: Lasso Regression, Random Forest, XGBoost  
  - Violence: Random Forest, XGBoost  
- **Evaluation**: 70/30 train-test split, normalization, hyperparameter tuning.  

## Key Findings
- 🌍 **Highest Risk Groups**: transgender & non-binary individuals, low-income populations, and multiply marginalized groups.  
- 📊 **Model Performance**: XGBoost outperformed other models.  
- 🔑 **Key Predictors**: gender identity, economic status, location comfort, and outness level.  
- 🌐 **Cross-National Variation**: discrimination risk differs significantly across countries.  

## Implications
- Insights highlight the urgent need for **policy interventions** protecting the most vulnerable groups.  
- Improved **representation of minority subgroups** in large-scale surveys will enhance predictive accuracy.  
- Future work should explore **longitudinal data** to track trends in LGBTIQ+ vulnerability.  

---

👩‍💻 Developed by **Wendi Xue**  
M.A. Computational Social Science, University of Chicago  
