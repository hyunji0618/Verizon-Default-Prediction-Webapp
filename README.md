# Verizon Customer Default Prediction & Contract Decisioning Web Application

A machine learning project that predicts customer default risk and translates model outputs into contract approval decisions and estimated financial impact. The project was designed as a data-driven decision support tool for Verizon staff to assess whether a customer should be approved for a phone contract during live customer interactions.

## Project Overview

Verizon’s phone contract business creates a lending-like risk: approving a customer who later defaults can lead to direct revenue loss, while rejecting a reliable customer can create missed profit and lost customer lifetime value.

This project addresses that tradeoff by building a default risk prediction workflow that:

* Predicts whether a customer is likely to default on a phone contract
* Compares multiple modeling approaches for imbalanced binary classification
* Prioritizes strong defaulter detection to reduce costly false negatives
* Evaluates business impact using revenue gained, losses avoided, and untapped potential
* Prototypes a real-time web application for store-level contract decision support

## Business Problem

The goal is not only to maximize model accuracy, but to support better contract approval decisions.

In this setting:

* True Positive: correctly identify a customer who would default and avoid contract loss
* False Positive: incorrectly reject a reliable customer and lose potential profit
* True Negative: correctly approve a customer who will pay
* False Negative: incorrectly approve a customer who will default and create financial loss

Because false negatives are costly, the final model focuses heavily on defaulter recall while still maintaining strong precision.

## Dataset

The dataset contains 24,833 customer contract records with 13 columns.

Key feature groups include:

* Contract timing: year, month, day
* Payment information: price, down payment, monthly payment, payment left, months due
* Customer risk indicators: credit score, payment type, default status
* Demographic fields used for analysis: age, gender

The target variable is `default`, where 1 indicates a customer defaulted and 0 indicates a non-defaulter.

## Exploratory Data Analysis

Key EDA findings:

* Dataset size: 24,833 rows and 13 columns
* No missing values across all columns
* Default rate: approximately 11.5%
* Credit score is an ordinal variable ranging from 0 to 8
* Payment fields contain outliers, including unusually high monthly payments and contract prices
* Age includes potential data quality issues, including 0 and 99+ values
* Payment type appears to capture different payment or contract plan categories

These findings shaped the modeling approach, especially the need to handle class imbalance, payment-related outliers, and risk-related financial tradeoffs.

## Modeling Approach

Several model families were considered:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine
* Neural Network
* XGBoost

The final selected model was XGBoost with oversampling.

XGBoost was selected because it performed well on imbalanced classification, handled non-linear relationships, showed robustness to outliers, and provided stronger defaulter detection than the neural network baseline.

## Model Comparison

| Model                     | Class         | Precision | Recall | F1-Score |
| ------------------------- | ------------- | --------- | ------ | -------- |
| Neural Network            | Non-Defaulter | 0.94      | 0.96   | 0.95     |
| Neural Network            | Defaulter     | 0.59      | 0.48   | 0.53     |
| XGBoost with Oversampling | Non-Defaulter | 0.95      | 0.85   | 0.90     |
| XGBoost with Oversampling | Defaulter     | 0.84      | 0.96   | 0.91     |

The neural network performed well on non-defaulters but struggled to identify defaulters. XGBoost with oversampling significantly improved performance on the defaulter class, which is the higher-risk group for the business decision.

## Final Model Assessment

Final XGBoost model performance:

* Overall accuracy: 91%
* Defaulter precision: 87%
* Defaulter recall: 96%
* True Positives: 4,190
* True Negatives: 3,777
* False Positives: 643
* False Negatives: 181

The model captures most defaulters while keeping the number of missed defaulters low. This is important because approving a customer who later defaults creates a larger financial loss than many standard classification errors.

## Business Impact Analysis

The model was evaluated not only with machine learning metrics, but also with financial assumptions around contract approval decisions.

Business assumptions used in the project:

* Customer base: 1 million applicants
* Profit per paying customer: $250
* Loss per default: $1,000
* Current approval rate: 80%
* Current default rate among approved customers: 5%

The model was used to estimate:

* Revenue from correctly approved paying customers
* Losses from incorrectly approved defaulters
* Untapped revenue from incorrectly rejected paying customers
* Loss avoidance from correctly rejected high-risk customers

This business framing connects model performance directly to revenue, loss avoidance, and contract approval strategy.

## Real-Time Decisioning App Prototype

The project includes a front-end prototype for a real-time customer default prediction tool.

Proposed workflow:

1. Customer fills out an initial form
2. Staff adds additional customer and contract details
3. The model returns an instant default risk prediction
4. Staff uses the prediction to support contract approval or denial decisions

## Demo

### App Demo Video

https://github.com/user-attachments/assets/7be8ee3e-71da-400a-b679-1e2ff52f08bf


