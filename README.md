# Default Prediction Flask App

This project is a Flask-based web application for predicting the default probability of a loan or payment based on user-provided data. The app uses a trained XGBoost model to make predictions and provides a simple and organized user interface with two pages for data input and a result page displaying the prediction.

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)

## Overview
The application collects input data in two steps:
1. **First Page**: Collects user details, such as name, payment type, credit score, age, and gender.
2. **Second Page**: Collects additional financial details, such as date, price, down payment, and monthly payment information.

The collected data is then fed into an XGBoost model to predict the default probability, which is displayed on the result page.

## Technologies Used
- **Flask**: A lightweight web application framework in Python.
- **XGBoost**: An efficient machine learning gradient boosting framework.
- **HTML/CSS**: For creating a responsive and clean user interface.
- **Pandas**: For data handling and manipulation.

## Usage
1. Navigate to the first page and fill in your personal details.
2. Proceed to the second page to enter financial details.
3. Submit the form to see the default probability prediction.


