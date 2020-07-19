from flask import Flask, redirect, request, render_template
import pickle
import re
from sklearn.naive_bayes import BernoulliNB
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

data = pickle.load(open('models/data.pkl', 'rb'));
models = pickle.load(open('models/model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
   return render_template("index.html")

@app.route("/", methods=['POST'])
def predict():
    location = request.form.get("location")
    year = request.form.get("year")
    distance = request.form.get("distance")
    fuel_type = request.form.get("fuel_type")
    transmission = request.form.get("transmission")
    owner_type = request.form.get("owner_type")
    mileage = request.form.get("mileage")
    engine = request.form.get("engine")
    power = request.form.get("power")
    seats = request.form.get("seats")
    model = request.form.get("model")

    d = np.array([location, year, distance, fuel_type, transmission, owner_type, mileage, engine, power, seats, model])
    d = {"d": d}
    d = pd.DataFrame(d)
    d = d.transpose()

    x = data.transform(d)
    ans = models.predict(x)
    
    
    return render_template("index.html", ans=str(round(ans[0], 2)))

if __name__ == '__main__':
   app.run(debug=True)
