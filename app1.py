from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
import pandas as pd
import joblib

app = Flask(__name__)

Lr = pickle.load(open('E:/Coding/Major/Flood-Prediction-Code/model.pkl', 'rb'))

Ur = pickle.load(open('E:/Coding/Major/Flood-Prediction-Code/model1.pkl', 'rb'))

model = pickle.load(open('E:/Coding/Major/Earthquake-Predictor-main/model2.pkl', 'rb'))

def fetch_weather_data(api_key, city_name):
    url = f'https://api.weatherapi.com/v1/forecast.json?key={api_key}&q={city_name}&alert=yes'
    response = requests.get(url)
    data = response.json()
    return data

# Define a function to preprocess weather data
def preprocess_weather_data(data):
    # Extract relevant features
    weather = {
        'temp_min': data['forecast']['forecastday'][0]['day']['mintemp_c'],
        'temp_max': data['forecast']['forecastday'][0]['day']['maxtemp_c'],
        'rain': data['current']['precip_mm'],
        'humidity': data['current']['humidity'],
        'pressure': data['current']['pressure_mb'],
        'wind_speed': data['current']['wind_kph'],
        'clouds': data['current']['cloud'] / 10
    }

    temp_min = weather['temp_min']
    temp_max = weather['temp_max']
    rain = weather['rain']
    humidity = weather['humidity']
    pressure = weather['pressure']
    wind_speed = weather['wind_speed']
    clouds = weather['clouds']

    input_data = pd.DataFrame({
        'temp_min': [temp_min],
        'temp_max': [temp_max],
        'rain': [rain],
        'wind_speed': [wind_speed],
        'humidity': [humidity],
        'pressure': [pressure],
        'clouds': [clouds]
    })
    return input_data

# Load ML model
cloud = joblib.load('E:/Coding/Major/CloudBurst-Prediction-Flask-main/model.joblib')  # Use the path to your model file



@app.route('/')
def home():
    return render_template('interface.html')

@app.route('/predict', methods=['POST'])
def predict():
    input1 = float(request.form.get('input1'))
    input2 = float(request.form.get('input2'))
    input3 = float(request.form.get('input3'))
    input4 = str(request.form.get('input4'))
    l = [[input1, input2, input3]]
    a = Lr.predict(l)
    b = Ur.predict(l)

    if input4 == "KERALA":
        if float(a) == 1:
            return render_template('floodkerala.html')
        else:
            return render_template('flood1.html')
    elif input4 == "UTTARAKHAND":
        if float(b) == 1:
            return render_template('flooduk.html')
        else:
            return render_template('flood1.html')
@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    data1 = int(float(request.form['a']))
    data2 = int(float(request.form['b']))
    data3 = int(float(request.form['c']))
    arr = np.array([[data1, data2, data3]])
    output = model.predict(arr)

    def to_str(var):
        return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]

    magnitude = output[0]

    if magnitude < 4:
        return render_template('prediction.html', p=to_str(magnitude), q='No')
    elif 4 <= magnitude < 6:
        return render_template('prediction.html', p=to_str(magnitude), q='Low')
    elif 6 <= magnitude < 8:
        return render_template('prediction.html', p=to_str(magnitude), q='Moderate')
    elif 8 <= magnitude < 9:
        return render_template('prediction.html', p=to_str(magnitude), q='High')
    elif magnitude >= 9:
        return render_template('prediction.html', p=to_str(magnitude), q='Very High')
    else:
        return render_template('prediction.html', p='N.A.', q='Undefined')

@app.route('/cloudpredict', methods=['POST'])
def cloudpredict():
    # Get city name from the form field
    city_name = request.form['city_name']
    if not city_name:
        return render_template('cloud.html', prediction_text='City name is required')

    # Fetch weather data
    weather_data = fetch_weather_data('7e1682484e304737beb50642240403', city_name)

    # Preprocess weather data
    processed_data = preprocess_weather_data(weather_data)

    # Make prediction
    prediction = cloud.predict(processed_data)
    prediction_text = 'Cloudburst' if prediction else 'No Cloudburst'

    # Pass weather data and prediction result to the template
    return render_template('cloud.html', weather_data=weather_data, city_name=city_name, prediction_text=prediction_text)



@app.route('/homepage1')
def homepage1():
    return render_template('homepage.html')

@app.route('/homepage2')
def homepage2():
    return render_template('index.html')

@app.route('/homepage3')
def homepage3():
    return render_template('cloud.html')


@app.route('/homepage5')
def homepage5():
    return render_template('map.html')
if __name__ == '__main__':
    app.run(debug=True)
