import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('cscore\models\modelo_scorecard.pkl', 'rb'))

@app.route('/')

def home():
    return render_template('plantilla.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Obtener datos del formulario y convertirlos a números
        person_income = float(request.form.get('person_income'))
        person_home_ownership = float(request.form.get('person_home_ownership'))
        person_emp_length = float(request.form.get('person_emp_length'))
        loan_intent = float(request.form.get('loan_intent'))
        loan_grade = float(request.form.get('loan_grade'))
        loan_amnt = float(request.form.get('loan_amnt'))
        loan_int_rate = float(request.form.get('loan_int_rate'))
        loan_percent_income = float(request.form.get('loan_percent_income'))
        cb_person_default_on_file = float(request.form.get('cb_person_default_on_file'))

 # Crear un DataFrame con los datos
        testDataframe = pd.DataFrame([[person_income, person_home_ownership, person_emp_length,
                                       loan_intent, loan_grade, loan_amnt, loan_int_rate,
                                       loan_percent_income, cb_person_default_on_file]],
                                     columns=["person_income", "person_home_ownership", "person_emp_length",
                                              "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
                                              "loan_percent_income", "cb_person_default_on_file"])

        # Realizar la predicción utilizando el modelo
        prediction = model.score(testDataframe)
        print("Resultado es:", prediction)

        # Devolver la respuesta, por ejemplo, la predicción
        return render_template('index.html', prediction_text='Resultado: {}'.format(prediction))

if __name__ == "__main__":
    app.run()