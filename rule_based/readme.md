Rule-Based Weather Activity Recommendation System

Prerequisites

Python Version
- Python 3.8 or 3.9 is required (the experta package supports these versions only).

Required Python Packages:
To install the necessary packages, run the following commands:

1. Experta (for rule-based reasoning):
   pip install experta

2. NumPy (version 1.x, compatible with experta):
   pip install numpy<2

3. Pandas (version 1.5.3, compatible with NumPy 1.x):
   pip install pandas==1.5.3

4. Geopy (for geolocation queries):
   pip install geopy

Important: Install the compatible versions of NumPy and Pandas before installing the Meteomatics API package.

Meteomatics API Parameters Required for Rule-Based Recommendations:

The following weather parameters from the Meteomatics API are required to enable the rule-based recommendation system:

- Minimum Temperature (24h, °C)
- Maximum Temperature (24h, °C)
- Wind Speed (24h, km/h)
- Wind Gusts (24h, km/h)
- Precipitation (24h, mm)
- UV Index (value)
- Weather Symbol (value) — This is the most critical parameter for the rule-based system.

Weather Symbol Explanation:
The weather symbol parameter from the Meteomatics API returns the current (momentary) weather condition (e.g., sunny, cloudy) for both day and night. The symbol is represented by a numerical value that corresponds to different weather conditions.

For a detailed description of available parameters, including the weather symbol values, please refer to the following links:
- Parameter Description: https://www.meteomatics.com/en/api/available-parameters/
- Weather Symbol Reference: https://www.meteomatics.com/en/api/available-parameters/weather-parameter/general-weather-state/#weather_symb

Example of Weather Symbol Parameter Usage:

<parameter name="weather_symbol_1h:idx">
    <location lat="50" lon="10">
        <value date="2024-09-25T00:00:00Z">103</value>  <!-- 103 represents rain at night -->
        <value date="2024-09-25T06:00:00Z">4</value>    <!-- 4 represents cloudy in the morning -->
    </location>
</parameter>


Example output of the code:

Enter the location name (e.g., Melbourne, Sydney): Melbourne
Coordinates for Melbourne: (-37.8142454, 144.9631732)
--------------------------------------------------
Minimum Temperature (24h): 9.5°C
Maximum Temperature (24h): 15.6°C
Wind Speed: 22.5 km/h
Wind Gusts (24h): 42.7 km/h
Precipitation (24h): 10.64 mm
UV Index: 0.0
Weather Symbol: 5.0
--------------------------------------------------
Recommended activities based on the current weather:
- Raincoat Shopping