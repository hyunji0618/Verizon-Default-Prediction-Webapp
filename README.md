# Default Prediction Flask App

This project is a Flask-based web application for predicting the default probability of a loan or payment based on user-provided data. The app uses a trained XGBoost model to make predictions and provides a simple and organized user interface with two pages for data input and a result page displaying the prediction.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Deployment](#deployment)
- [Screenshots](#screenshots)

## Overview
The application collects input data in two steps:
1. **First Page**: Collects user details, such as name, payment type, credit score, age, and gender.
2. **Second Page**: Collects additional financial details, such as date, price, down payment, and monthly payment information.

The collected data is then fed into an XGBoost model to predict the default probability, which is displayed on the result page.

## Features
- **Simple and User-Friendly Forms**: Collect user inputs in two steps.
- **Prediction Using XGBoost**: Uses a trained XGBoost model to calculate the probability of default.
- **Responsive Design**: Clean and organized UI with a consistent design using HTML and CSS.
- **Easy Deployment**: The app is ready to be deployed on platforms like Heroku, AWS, or Google Cloud.

## Technologies Used
- **Flask**: A lightweight WSGI web application framework in Python.
- **XGBoost**: A powerful and efficient gradient boosting framework.
- **HTML/CSS**: For creating a responsive and clean user interface.
- **Pandas**: For data handling and manipulation.
