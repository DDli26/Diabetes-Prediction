import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
scaler=pickle.load(open("Models/standard_scaler.pkl", 'rb'))
logistic_model=pickle.load(open("Models/logistic_model.pkl", 'rb'))

@app.route("/")
def home_page():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method=='POST':
        x=[] #this will store all the values for the variables and based on that our model will make predictions
        for key in request.form.keys():
            x.append(int(request.form.get(key)))
        x_scaled=scaler.transform([x])
        
        diabetes_probability=logistic_model.predict_proba(x_scaled)
        diabetes_probability=diabetes_probability[0][1]*100
        return render_template("result.html", percentage=diabetes_probability)
    return render_template('home.html')


if __name__=='__main__':
    print('hey')
    app.run()