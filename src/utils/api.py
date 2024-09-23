from geopy.geocoders import Nominatim
import meteomatics.api as weatherAPI
from datetime import timedelta


def get_coordinates_from_city(city_name):
    geolocator = Nominatim(user_agent="weather_chatbot")
    location = geolocator.geocode(city_name)
    
    if location:
        return location.latitude, location.longitude
    else:
        return None, None  # Return None if the city is not found

#import pandas as pd

def query_time_series_data(coordinates, startdate, enddate, username, password):
    """
    Queries time series data from the API using the provided parameters and returns a pandas DataFrame.

    Args:
        coordinates (str): Coordinates for the data query (e.g., latitude/longitude).
        startdate (str): The start date for the data query (format: YYYY-MM-DD).
        enddate (str): The end date for the data query (format: YYYY-MM-DD).
        interval (str): The time interval (e.g., hourly, daily).
        parameters (list): List of parameters to query (e.g., temperature, humidity).
        username (str): API username.
        password (str): API password.
        model (str): The data model to use for the query (default is "mix").

    Returns:
        pd.DataFrame: DataFrame containing the time series data from the API.
    """
    interval = timedelta(days=1)
    # Weather Parameters
    parameters = ['t_max_2m_24h:C', 't_min_2m_24h:C', 'wind_speed_10m:ms', 'precip_24h:mm', 'msl_pressure:hPa','uv:idx']
    #print(coordinates,startdate,enddate,interval,parameters,username,password)
    try:
        # Make the API call
        df = weatherAPI.query_time_series(
            coordinate_list=coordinates,
            startdate=startdate,
            enddate=enddate,
            interval=interval,
            parameters=parameters,
            username=username,
            password=password,
            model= "mix"
        )
        return df
    except Exception as e:
        print(f"Error occurred while querying the time series data: {e}")
        return None
