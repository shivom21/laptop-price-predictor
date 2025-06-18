from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model and features
model, feature_cols = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form['company']
    typename = request.form['typename']
    ram = int(request.form['ram'])
    memory_type = request.form['memory']
    age = int(request.form['age'])

    memory_code = {'HDD': 0, 'SSD': 1, 'Other': 2}.get(memory_type, 2)

    input_dict = {
        'Ram': ram,
        'Memory': memory_code,
        'Age': age,
    }

    for col in feature_cols:
        if col.startswith('Company_'):
            input_dict[col] = 1 if col == f'Company_{company}' else 0
        elif col.startswith('TypeName_'):
            input_dict[col] = 1 if col == f'TypeName_{typename}' else 0

    for col in feature_cols:
        if col not in input_dict:
            input_dict[col] = 0

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    prediction = np.round(prediction)

    return render_template('index.html', prediction_text=f"₹ {prediction:,.0f}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
