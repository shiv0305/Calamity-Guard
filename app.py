from flask import Flask, jsonify, render_template, request
import pickle
import numpy as np
import requests
import pandas as pd
import joblib

app = Flask(__name__)

Lr = pickle.load(open('E:/Coding/Major/Flood-Prediction-Code/model.pkl', 'rb'))

Ur = pickle.load(open('E:/Coding/Major/Flood-Prediction-Code/model1.pkl', 'rb'))

model = pickle.load(open('E:/Coding/Major/Earthquake-Predictor-main/model2.pkl', 'rb'))

meteor_showers = pd.read_csv('E:/Coding/Major/Predict-meteor-showers-using-Python-main/data/meteorshowers.csv')
moon_phases = pd.read_csv('E:/Coding/Major/Predict-meteor-showers-using-Python-main/data/moonphases.csv')
constellations = pd.read_csv('E:/Coding/Major/Predict-meteor-showers-using-Python-main/data/constellations.csv')
cities = pd.read_csv('E:/Coding/Major/Predict-meteor-showers-using-Python-main/data/cities.csv')

# Define month mapping
months = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5,
    'june': 6, 'july': 7, 'august': 8, 'september': 9,
    'october': 10, 'november': 11, 'december': 12
}

# Map month names to numbers and convert dates
meteor_showers['bestmonth'] = meteor_showers['bestmonth'].map(months)
meteor_showers['startmonth'] = meteor_showers['startmonth'].map(months)
meteor_showers['endmonth'] = meteor_showers['endmonth'].map(months)
moon_phases['month'] = moon_phases['month'].map(months)
constellations['bestmonth'] = constellations['bestmonth'].map(months)

# Convert to datetime
meteor_showers['startdate'] = pd.to_datetime(
    2024 * 10000 + meteor_showers['startmonth'] * 100 + meteor_showers['startday'],
    format='%Y%m%d')
meteor_showers['enddate'] = pd.to_datetime(
    2024 * 10000 + meteor_showers['endmonth'] * 100 + meteor_showers['endday'],
    format='%Y%m%d')
moon_phases['date'] = pd.to_datetime(
    2024 * 10000 + moon_phases['month'] * 100 + moon_phases['day'],
    format='%Y%m%d')

# Hemisphere and moon phase mappings
hemispheres = {'northern': 0, 'southern': 1, 'northen, southern': 3}
meteor_showers['hemisphere'] = meteor_showers['hemisphere'].map(hemispheres)
constellations['hemisphere'] = constellations['hemisphere'].map(hemispheres)

phases = {'new moon': 0, 'third quarter': 0.5, 'first quarter': 0.5, 'full moon': 1.0}
moon_phases['percentage'] = moon_phases['moonphase'].map(phases)
moon_phases = moon_phases.drop(['month', 'day', 'moonphase', 'specialevent'], axis=1)

# Fill missing moon phase data
last_phase = 0
for index, row in moon_phases.iterrows():
    if pd.isnull(row['percentage']):
        moon_phases.at[index, 'percentage'] = last_phase
    else:
        last_phase = row['percentage']

# Drop unnecessary columns
meteor_showers = meteor_showers.drop(['startmonth', 'startday', 'endmonth', 'endday', 'hemisphere'], axis=1)
constellations = constellations.drop(['besttime'], axis=1)

# Prediction function
def predict_best_meteor_shower_viewing(city):
    meteor_shower_string = ""

    if city not in cities['city'].values:
        return f"Unfortunately, {city} isn't available for a prediction at this time."

    latitude = cities.loc[cities['city'] == city, 'latitude'].iloc[0]
    meteor_shower_string += f"Latitude of {city}: {latitude}\n"
    
    constellation_list = constellations.loc[
        (constellations['latitudestart'] >= latitude) & 
        (constellations['latitudeend'] <= latitude), 'constellation'
    ].tolist()
    meteor_shower_string += f"Constellations visible from {city}: {constellation_list}\n"

    if not constellation_list:
        return f"Unfortunately, there are no meteor showers viewable from {city}."

    for constellation in constellation_list:
        meteor_shower_data = meteor_showers.loc[meteor_showers['radiant'] == constellation]
        meteor_shower_string += f"Meteor showers for constellation {constellation}: {meteor_shower_data}\n"

        if meteor_shower_data.empty:
            continue

        meteor_shower = meteor_shower_data['name'].iloc[0]
        meteor_shower_startdate = meteor_shower_data['startdate'].iloc[0]
        meteor_shower_enddate = meteor_shower_data['enddate'].iloc[0]

        moon_phases_list = moon_phases.loc[
            (moon_phases['date'] >= meteor_shower_startdate) & 
            (moon_phases['date'] <= meteor_shower_enddate)
        ]

        if moon_phases_list.empty:
            continue

        if meteor_shower == "Chang'e":
            best_moon_date = moon_phases_list.loc[moon_phases_list['percentage'].idxmax()]['date']
            meteor_shower_string += (
                f"Though the Moon will be bright, {meteor_shower}'s meteor shower is best seen if you look "
                f"towards the {constellation} constellation on {best_moon_date.strftime('%B %d, %Y')}.\n"
            )
        else:
            best_moon_date = moon_phases_list.loc[moon_phases_list['percentage'].idxmin()]['date']
            meteor_shower_string += (
                f"{meteor_shower} is best seen if you look towards the {constellation} constellation on "
                f"{best_moon_date.strftime('%B %d, %Y')}.\n"
            )

    return meteor_shower_string if meteor_shower_string else f"No meteor showers are available for viewing from {city} at this time."



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

@app.route('/metpredict', methods=['POST'])
def metpredict():
    data = request.get_json()
    city = data.get('city')
    if city:
        prediction = predict_best_meteor_shower_viewing(city)
        return jsonify({'message': prediction})
    else:
        return jsonify({'message': 'Please provide a valid city.'}), 400


@app.route('/homepage1')
def homepage1():
    return render_template('homepage.html')

@app.route('/homepage2')
def homepage2():
    return render_template('index.html')

@app.route('/homepage3')
def homepage3():
    return render_template('cloud.html')

@app.route('/homepage4')
def homepage4():
    return render_template('meteor.html')

@app.route('/homepage5')
def homepage5():
    return render_template('map.html')
if __name__ == '__main__':
    app.run(debug=True)
