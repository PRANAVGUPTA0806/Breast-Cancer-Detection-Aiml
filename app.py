from flask import Flask, render_template, url_for ,request

from static.breast_cancer import X_test, Y_test
app = Flask(__name__)

import pickle
model=pickle.load(open("models/breast_cancer.pkl","rb"))
with open("models/model4.pkl", "rb") as f:
    model4= pickle.load(f)
model5=pickle.load(open("models/insurance.pkl","rb"))
import numpy as np


@app.route('/')
def index2():
    return render_template('index2.html')

@app.route('/options')
def options():
    return render_template('bc_index4.html')

@app.route('/form1')
def form1():
    return render_template('bc_index2.html')

@app.route('/form')
def form():
    return render_template('bc_index3.html')

@app.route('/insurance')

def insurance():
    return render_template('insurance.html')

@app.route('/predict6', methods=['POST','GET'])
def predict6():

    features= request.form.to_dict()
    data_array = np.array(list(features.values()), dtype=float)
    data_array=data_array.reshape(1, -1)  
    prediction = model5.predict(data_array).item()
    if prediction < 0:
        return render_template('amount.html', prediction='Error calculating Amount!')
    else:
        formatted_pred = 'Expected amount is {0:.3f}'.format(prediction)
        return render_template('amount.html',prediction=formatted_pred)
    
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        data_array = np.array(list(form_data.values()), dtype=float)
        data_array=data_array.reshape(1, -1)  
        prediction = model.predict(data_array)
        n=""
        # Return the prediction
        if(prediction[0]==0):
            return render_template('B.html')
        else:
            return render_template('B.html')

@app.route('/predict2', methods=['POST'])
def predict2():
    if request.method == 'POST':
        # Extract input data from form
        csv_input = request.form['csv-input']
        
        # Convert comma-separated string to array of floats
        features = list(map(float, csv_input.split(',')))
        data_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(data_array)
        if(prediction[0]==0):
            return render_template('M.html')
        else:
            return render_template('B.html')
        
    
       
if __name__ == '__main__':
    app.run()
    