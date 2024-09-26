import datetime as dt
import meteomatics.api as api
from geopy.geocoders import Nominatim
from test_code.func import rec_activity  # Import the rule-based recommendation function

# Meteomatics API credentials
username = "rmituniversity_haruethaithanasan_visutinee"
password = "vTe24kC2Yy"

# Initialize the Nominatim geolocator from Geopy
geolocator = Nominatim(user_agent="weather_app")

# Function to get coordinates for a location using Geopy
def get_coordinates(location):
    try:
        location_data = geolocator.geocode(location)
        if location_data:
            return (location_data.latitude, location_data.longitude)
        else:
            return None
    except Exception as e:
        print(f"Error fetching coordinates: {e}")
        return None

# Function to get weather data using the Meteomatics API
def get_weather_data(coordinates):
    # Weather parameters to query
    parameters1 = [
        't_min_2m_24h:C', 't_max_2m_24h:C', 'wind_gusts_10m_24h:kmh',
        'msl_pressure:hPa', 'precip_24h:mm', 'uv:idx'
    ]
    
    # Lock in the current time for the weather forecast
    current_time = dt.datetime.now(dt.timezone.utc)
    startdate = current_time
    enddate = current_time
    interval = dt.timedelta(hours=0)

    try:
        df = api.query_time_series(
            coordinate_list=[coordinates],
            startdate=startdate, 
            enddate=enddate, 
            interval=interval, 
            parameters=parameters1, 
            username=username, 
            password=password, 
            model='mix'
        )

        if not df.empty:
            for index, row in df.iterrows():
                t_max = row['t_max_2m_24h:C']
                t_min = row['t_min_2m_24h:C']
                wind_gust = row['wind_gusts_10m_24h:kmh']
                msl_pressure = row['msl_pressure:hPa']
                precip = row['precip_24h:mm']
                uv_idx = row['uv:idx']

                print("--------------------------------------------------")
                print(f"Minimum Temperature (24h): {t_min}°C")
                print(f"Maximum Temperature (24h): {t_max}°C")
                print(f"Wind Gusts (24h): {wind_gust} km/h")
                print(f"MSL Pressure: {msl_pressure} hPa")
                print(f"Precipitation (24h): {precip} mm")
                print(f"UV Index: {uv_idx}")
                print("--------------------------------------------------")

                # Change here to adjust arguements
                # recommendations = rec_activity(t_max, wind_gust, uv_idx)
                recommendations = rec_activity(t_max)

                
                print("Recommended activities based on the current weather:")
                for activity in recommendations:
                    print(f"- {activity}")
        else:
            print("No data available.")

    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None

if __name__ == "__main__":
    while True:
        location = input("Enter the location name (or type 'exit' to terminate): ").strip()
        
        if location.lower() == "exit":
            print("Program terminated.")
            break
        
        coordinates = get_coordinates(location)
        
        if coordinates:
            print(f"Coordinates for {location}: {coordinates}")
            get_weather_data(coordinates)
        else:
            print(f"Location '{location}' not found.")
