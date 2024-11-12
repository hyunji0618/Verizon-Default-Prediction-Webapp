from flask import Flask, render_template, request, redirect, url_for
import xgboost as xgb
import pandas as pd
from xgboost_model import bst
import os

app = Flask(__name__)

training_columns = ['year', 'month', 'day', 'price', 'downpmt', 'monthdue', 'payment_left',
                    'monthly_payment', 'credit_score', 'age', 'gender_2', 'pmttype_3', 'pmttype_4', 'pmttype_5']

# Global dictionary to hold form data across pages
form_data = {}

@app.route('/', methods=['GET', 'POST'])
def first_page():
    if request.method == 'POST':
        # Collect data from the first page
        form_data['pmttype'] = int(request.form.get('pmttype'))
        form_data['credit_score'] = int(request.form.get('credit_score'))
        form_data['age'] = int(request.form.get('age'))
        form_data['gender'] = int(request.form.get('gender'))
        return redirect(url_for('second_page'))
    return render_template('first_page.html')

@app.route('/second_page', methods=['GET', 'POST'])
def second_page():
    if request.method == 'POST':
        # Collect data from the second page
        form_data['year'] = int(request.form.get('year'))
        form_data['month'] = int(request.form.get('month'))
        form_data['day'] = int(request.form.get('day'))
        form_data['price'] = float(request.form.get('price'))
        form_data['downpmt'] = float(request.form.get('downpmt'))
        form_data['monthdue'] = float(request.form.get('monthdue'))
        form_data['payment_left'] = float(request.form.get('payment_left'))
        form_data['monthly_payment'] = float(request.form.get('monthly_payment'))

        # Prepare the input data for prediction
        input_data = pd.DataFrame([form_data])
        
        # Convert categorical features to dummy variables (same as in training)
        input_data = pd.get_dummies(input_data, columns=['gender', 'pmttype'], drop_first=True)

        for column in training_columns:
            if column not in input_data:
                input_data[column] = 0  # Add missing columns with a value of 0

        # Reorder columns to match the training data
        input_data = input_data[training_columns]

        # Convert input data into DMatrix
        dmatrix = xgb.DMatrix(input_data)

        # Predict using the XGBoost model
        y_pred = bst.predict(dmatrix)[0]
        probability = round(y_pred * 100, 4)

        return render_template('result.html', probability=probability)
    return render_template('second_page.html')

#if __name__ == '__main__':
#    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

if __name__ == '__main__':
    app.run(debug=True)

