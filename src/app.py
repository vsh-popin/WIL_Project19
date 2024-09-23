from chatterbot import ChatBot
from flask import Flask, session, render_template, request
import utils.regex as reg
import utils.api as api
import utils.model as weather_model
from datetime import datetime, timezone
import dateparser
import configparser
import pandas as pd
import torch
#from sklearn.preprocessing import MinMaxScaler
from models.predict_model import CNNWeatherPredictor
app = Flask(__name__, static_folder="templates/static")

chatbot = ChatBot('MyBot')

# Initialize the parser and read the config file
config = configparser.ConfigParser()
config.read('config.ini')
#app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default_fallback_key')
app.secret_key = config.get('FLASK', 'secret_key', fallback='default_fallback_key')
# Access the credentials
username = config.get('API', 'username')
password = config.get('API', 'password')

@app.route("/")
def home():
    return render_template("index.html")

@app.before_request 
def before_request(): # Initialize the session variables
    if 'city' not in session:
        session['city'] = None
    if 'date' not in session:
        session['date'] = None
        session['date_text'] = None

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_text = request.form["msg"]
    
    # Check if user input contains a valid date
    date, msg = extract_time_from_input(user_text)

    # Check for weather-related or city-related queries
    city = reg.find_australian_city_in_input(user_text)
    
    # Debugging prints
    print("date=", date)
    print("city=", city)
    
    # If both city and date are provided in the same input
    if city and date:
        session['city'] = city
        session['date'] = date
        session['date_text'] = msg

        response = get_weather(city, date)
        # Reset session variables after use
        reset_session()
        return response

    # If only city is detected
    elif city:
        session['city'] = city
        if session['date']:
            response = get_weather(city, session['date'])
            # Reset after use
            reset_session()
        else:
            return f"Cool! I have got [{city}] down. When do you need the weather for? You can say something like 'tomorrow' or 'in 3 days'."

    # If only a date is detected
    elif date:
        session['date'] = date
        session['date_text'] = msg
        if session['city']:
            response = get_weather(session['city'], date)
            # Reset after use
            reset_session()
        else:
            return f"I have got the time [{session['date_text']}]. Now, let me know which city in Australia you would like the weather for."

    # If neither city nor date is found in the input
    else:
        if session['city']:
            return f"You're asking about the weather in [{session['city']}]. Now, when do you need the forecast? Maybe 'tomorrow' or 'in a few days'?"
        elif session['date']:
            return f"I've noted [{session['date_text']}]. Could you let me know which Australian city you'd like the weather for?"
        else:
            return "Could you tell me both a city in Australia and a time period you're interested in?"

    return response

# Function to call Meteomatics API to get the weather for a city
def get_weather(city, date=None):
    lat, lon = api.get_coordinates_from_city(city)
    
    if lat is None or lon is None:
        return f"Sorry, I couldn't find the location for {city}."
    print("get_weather=====") 
    print(lat,lon)
    # Coordinates as a tuple
    coordinates = [(lat, lon)]
    
    # If date is a string, convert it to a datetime object in UTC timezone
    if isinstance(date, str):
        try:
            # Convert string date (format "YYYY-MM-DD") to a timezone-aware datetime object in UTC
            date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return "Invalid date format. Please use YYYY-MM-DD."

    # If no date is provided, use the current UTC date and time
    if date is None:
        date = datetime.now(timezone.utc)
    
    # Define the enddate as the given or current date and calculate the startdate (90 days before)
    enddate = date
    startdate= date #FIXME
    #startdate = enddate - timedelta(days=89)  # Subtract 90 days from enddate
    #print("enddate=",enddate)
    #print("startdate=",startdate)


    #try:
        # Call the Meteomatics API for a time series
        #df = api.query_time_series_data(coordinates, startdate, enddate, username, password)
       
    df = pd.read_csv("/Users/popin/Documents/GitHub/Chatbot/data/mockdata.csv", header=None)
    if df is None:
        return "Sorry, I couldn't retrieve the data."
    print(df)
    

    # Load scaler and model
    scaler = weather_model.load_scaler('./models/scaler.pkl')
    data_scaled = weather_model.scale_data(scaler,df)
    
    #model = weather_model.load_model('./models/CNN_Model_12Sep.pth')
    model= torch.load('./models/CNN_Model_12Sep.pth', map_location=torch.device('cpu'))
    model.eval()
 
    # Make prediction
    predicted_values = weather_model.make_prediction(model, data_scaled, scaler)
    #print(predicted_values)
    
    weather_report = generate_weather_report(predicted_values, city, date)

    # # Now you can print or return the weather report
    # for report in weather_report:
    #     print(report)
    
    return weather_report
    #except Exception as e:
       # return f"Sorry, I couldn't retrieve the weather for {city}. Error: {str(e)}"
    
def generate_weather_report(predicted_values, city, date):

    weather_report = []
    
    for i, row in enumerate(predicted_values[:1]):  # Only consider the first row if multiple rows exist
        max_temp = row[0]
        min_temp = row[1]
        wind_speed = row[2]
        pressure = row[3]
        precipitation = row[4]
        uv_index = row[5]

        report = (
            f"Weather in {city} on {date.strftime('%Y-%m-%d')}:\n"
            f"Max temp: {max_temp:.2f}°C,\n"
            f"Min temp: {min_temp:.2f}°C,\n"
            f"Wind speed: {wind_speed:.2f} m/s,\n"
            f"Pressure: {pressure:.2f} hPa,\n"
            f"Precipitation: {precipitation:.2f} mm,\n"
            f"UV index: {uv_index:.2f}."
        )
        
        weather_report.append(report)
    
    return weather_report

def reset_session():
    """Reset session variables."""
    session['city'] = None
    session['date'] = None
    session['date_text'] = None
def extract_time_from_input(user_input):

    processed_date, msg = reg.preprocess_relative_dates(user_input)
 
    print(f"Debug: Processed input: {processed_date}")
    
    # Parse the input with dateparser
    parsed_date = dateparser.parse(processed_date, settings={
        'PREFER_DATES_FROM': 'future',
        'PREFER_DAY_OF_MONTH': 'first',
        'RETURN_AS_TIMEZONE_AWARE': False
    })
    
    # Check if parsing succeeded and return the formatted date
    if parsed_date:
        # Date Definition
        return parsed_date.strftime('%Y-%m-%d'), msg
    else:
        print("Debug: No date found.")
        return None, "Sorry, Could you provide period time from your input."

if __name__ == "__main__":
    app.run(debug=True)
