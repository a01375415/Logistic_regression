# Customer Churn Prediction using Logistic Regression

## Introduction
Customer churn, or customer attrition, is a critical concern for businesses in various industries. It refers to the loss of customers or subscribers to a service. Predicting and understanding customer churn can help companies proactively retain valuable customers. In this assignment, you will work with a "Customer Churn Dataset" to build a logistic regression model for predicting customer churn. 

## Dataset Description
- **Dataset Name:** Customer Churn Dataset
- **Data Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Variables:** The dataset contains several features related to customer behavior and demographics and a binary target variable indicating whether a customer churned (1) or not (0).

## Assignment Objectives
1. **Data Preprocessing:** Clean and prepare the dataset for analysis. This includes handling missing values, encoding categorical variables, and scaling if necessary.
2. **Explore Data:** Conduct exploratory analysis to gain insights into the dataset. Visualize the data using appropriate plots and summarize key statistics.
3. **Feature Selection:** Identify relevant features for predicting customer churn. You can use techniques like correlation analysis.
4. **Logistic Regression Model:** Build a logistic regression model to predict customer churn. Split the dataset into training and testing sets for model evaluation.
5. **Model Evaluation:** Evaluate the model's performance using appropriate metrics such as accuracy, precision, recall, F1-score, and ROC AUC. Interpret the results.
6. **Addressing Class Imbalance:** If the dataset exhibits class imbalance, implement techniques to address this issue for model construction.

## Instructions
### Data Preparation
- Split the dataset into a training set (70-80%) and a testing set (20-30%) for model evaluation.

### Exploratory Data Analysis (EDA)
- Conduct exploratory analysis to understand the dataset.
- Create visualizations (e.g., histograms, box plots, scatter plots) to explore relationships between features and the target variable.
- Summarize key statistics and observations.

### Feature Selection
- Identify which features are likely to be most informative for predicting customer churn.
- Consider using correlation analysis.

### Logistic Regression Modeling
- Build a logistic regression model using the training data.
- Tune hyperparameters if necessary.
- Make predictions on the test set.

### Model Evaluation
- Evaluate the model's performance on the test set using appropriate evaluation metrics.
- Discuss the implications of the evaluation results for the business.

### Addressing Class Imbalance (Optional)
- If class imbalance is observed, implement strategies to mitigate it.
- Reevaluate the model's performance after addressing class imbalance.
