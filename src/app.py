from chatterbot import ChatBot
from flask import Flask, session, render_template, request, jsonify
import utils.regex as reg
import utils.api as api
import utils.model as weather_model
from datetime import datetime,timedelta, timezone
import dateparser
import configparser
#import pandas as pd
import torch
from models.predict_model import TCNWeatherPredictor,TemporalBlock
import models.func as rb

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
    session['trip'] = None
    #session['trip_detail'] = None

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_text = request.form["msg"]
    
    # Check if user input contains a valid date
    date, msg = extract_time_from_input(user_text)

    # Check for weather-related or city-related queries
    city = reg.find_australian_city_in_input(user_text)
    trip_check = reg.is_suggestion_trip_related(user_text)
    # Debugging prints
    print("date=", session['date'])
    print("city=", session['city'])
    print("trip=", trip_check)
    print("Detail", session['trip_detail'])
    # If both city and date are provided in the same input

    if city and date:
        session['city'] = city
        session['date'] = date
        session['date_text'] = msg
        session['trip'] = trip_check

        response,job_type = get_weather(city, date)
        # Reset session variables after use
        reset_session()
                # Ask a follow-up question
        if trip_check:
            response,job_type = suggest_trip_based_on_weather()

        return jsonify({"message": response,"job_type": job_type})

    # If only city is detected
    elif city:
        session['city'] = city
        if session['trip'] == None:   session['trip'] = trip_check
        if session['date']:
            response,job_type = get_weather(city, session['date'])
            # Reset after use
            if trip_check:
                response,job_type = suggest_trip_based_on_weather()
            reset_session()
        else:
            response,job_type = f"Cool! I have got [{city}] down. When do you need the weather for? You can say something like 'tomorrow' or 'in 3 days'.",0
            return jsonify({"message": response,"job_type": job_type})
        
    # If only a date is detected
    elif date:
        session['date'] = date
        session['date_text'] = msg
        if session['trip'] == None:   session['trip'] = trip_check
        if session['city']:
            response,job_type = get_weather(session['city'], date)
            if trip_check:
                response,job_type = suggest_trip_based_on_weather()
            # Reset after use
            reset_session()
        else:
            response,job_type = f"I have got the time [{session['date_text']}]. Now, let me know which city in Australia you would like the weather for.",0
            return jsonify({"message": response,"job_type": job_type})

    # If neither city nor date is found in the input
    else:
        if session['trip'] == None:   session['trip'] = trip_check
        if session['city']:
            response,job_type = f"You're asking about the weather in [{session['city']}]. Now, when do you need the forecast? Maybe 'tomorrow' or 'in a few days'?",0
            return jsonify({"message": response,"job_type": job_type})
        elif session['date']:
            response,job_type = f"I've noted [{session['date_text']}]. Could you let me know which Australian city you'd like the weather for?",0
            return jsonify({"message": response,"job_type": job_type})
        elif session['trip'] and session['trip_detail']:
            response,job_type = suggest_trip_based_on_weather()
            return jsonify({"message": response,"job_type": job_type})
        else:
            response,job_type = "Could you tell me both a city in Australia and a time period you're interested in?",0
            return jsonify({"message": response,"job_type": job_type})

    return jsonify({"message": response,"job_type": job_type})

# Function to call Meteomatics API to get the weather for a city
def get_weather(city, date=None):
    lat, lon = api.get_coordinates_from_city(city)
    
    if lat is None or lon is None:
        return f"Sorry, I couldn't find the location for {city}."
    
    #print(lat,lon)
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
    startdate = enddate - timedelta(days=89)  # Subtract 90 days from enddate
    #print("enddate=",enddate)
    #print("startdate=",startdate)

    try:
        # Call the Meteomatics API for a time series
        df = api.query_time_series_data(coordinates, startdate, enddate, username, password)
        #df = pd.read_csv("/Users/popin/Documents/GitHub/Chatbot/data/mockdata.csv", header=None) #FIXME
        if df is None:
            return "Sorry, I couldn't retrieve the data."
        #print(df)
    
        # Load scaler and model
        scaler = weather_model.load_scaler('./models/scaler.pkl')
        data_scaled = weather_model.scale_data(scaler,df)
        
        model= torch.load('./models/tcn_6_attr.pth', map_location=torch.device('cpu'))
        model.eval()

        predicted_values = weather_model.make_prediction(model, data_scaled)
        
        weather_report = generate_weather_report(predicted_values, city, date)

        return weather_report,1
    except Exception as e:
        return f"Sorry, I couldn't retrieve the weather for {city}. Error: {str(e)}",0
    
def generate_weather_report(predicted_values, city, date):

    weather_report = []
    #max_temp,min_temp,wind_speed,precipitation,pressure,uv_index
    for i, row in enumerate(predicted_values[:1]):  # Only consider the first row if multiple rows exist
        # Extract and format each value based on your requirements
        max_temp = round(float(row[0]))  
        min_temp = round(float(row[1]))  
        wind_speed = round(float(row[2]), 1) 
        precipitation = round(float(row[3]), 1) 
        pressure = round(float(row[4])) 
        uv_index = round(float(row[5]))  

        # # Create the weather report
        weather_report = [
            max_temp,    
            min_temp,     
            wind_speed,  
            precipitation,  
            pressure,    
            uv_index
        ]
        #weather_report = row.tolist() 
        weather_report.append(str(city))
        weather_report.append(str(date.strftime('%A,%d %B %Y')))
        
        #trip_detail = row.tolist()  # Convert to list if it's a NumPy or Pandas object

        # Ensure city is a string
        #trip_detail.append(str(city))
        #weather_report = trip_detail
        # Store the array in session
        session['trip_detail'] = weather_report
    
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

def suggest_trip_based_on_weather():

    #print("trip_detail3=", session['trip_detail'])
 
    if session['trip_detail']:
        session['trip'] = None
        #trip_detail = session['trip_detail']
        
        #rec = rb.rec_activity(trip_detail[0])
        #print(rec)
        #return "test"
        #trip_detail = session['trip_detail']
        #return f"{rec}"
        #Activity name,Description,Ideal temp,Location
        #activity_details = ["Activity name", "Description", "Ideal temp", "Location"]
        activity_details = rb.get_random_static()
        return activity_details,2
    else:
        return "I don't have enough information to suggest a trip. Please ask for the weather first.",0

if __name__ == "__main__":
    app.run(debug=True)
