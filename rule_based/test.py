import datetime as dt
import meteomatics.api as api
from geopy.geocoders import Nominatim

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
        't_min_2m_24h:C', 't_max_2m_24h:C', 'wind_speed_10m:kmh',
        'wind_gusts_10m_24h:kmh', 'precip_24h:mm', 'uv:idx', 'weather_symbol_24h:idx'
    ]
    
    # Time period for weather forecast
    enddate = dt.datetime.now(dt.timezone.utc)  # Use timezone-aware UTC datetime
    startdate = enddate - dt.timedelta(days=0)
    interval = dt.timedelta(hours=12)

    # Query weather data using the Meteomatics API
    try:
        # Pass the coordinates as a list of tuples
        df = api.query_time_series(
            coordinate_list=[coordinates],  # Wrap the coordinates tuple in a list
            startdate=startdate, 
            enddate=enddate, 
            interval=interval, 
            parameters=parameters1, 
            username=username, 
            password=password, 
            model='mix'
        )

        # Extract the necessary weather parameters
        if not df.empty:
            for index, row in df.iterrows():
                t_max = row['t_max_2m_24h:C']
                wind_speed = row['wind_speed_10m:kmh']
                uv_index = row['uv:idx']
                weather_symbol = row['weather_symbol_24h:idx']

                print("--------------------------------------------------")
                print(f"Minimum Temperature (24h): {row['t_min_2m_24h:C']}°C")
                print(f"Maximum Temperature (24h): {t_max}°C")
                print(f"Wind Speed: {wind_speed} km/h")
                print(f"Wind Gusts (24h): {row['wind_gusts_10m_24h:kmh']} km/h")
                print(f"Precipitation (24h): {row['precip_24h:mm']} mm")
                print(f"UV Index: {uv_index}")
                print(f"Weather Symbol: {weather_symbol}")
                print("--------------------------------------------------")

                # Run the rule-based activity recommender
                # recommendations = rec_activity(t_max, wind_speed, uv_index, weather_symbol)
                
                # Display the recommended activity
                # print("Recommended activities based on the current weather:")
                # for activity in recommendations:
                #     print(f"- {activity}")
        else:
            print("No data available.")

    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None

# Main program logic
if __name__ == "__main__":
    # Ask the user for the location
    location = input("Enter the location name (e.g., Melbourne, Sydney): ").strip()

    # Get coordinates for the given location using Geopy
    coordinates = get_coordinates(location)
    
    if coordinates:
        print(f"Coordinates for {location}: {coordinates}")
        
        # Query the Meteomatics API for weather data
        weather_data = get_weather_data(coordinates)
        
        if weather_data is not None:
            print("Weather data:")
            print(weather_data)
        else:
            print("No weather data available.")
    else:
        print(f"Location '{location}' not found.")
