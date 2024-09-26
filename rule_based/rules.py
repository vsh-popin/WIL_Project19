from experta import *

# Define the knowledge engine for weather-based activity recommendations
class WeatherRecommender(KnowledgeEngine):
    
    # Rule for recommending "Beach" based on hot weather, calm winds, and moderate UV index
    @Rule(
        Fact(t_max=P(lambda x: x > 25)),  # Hot weather
        Fact(wind_speed=P(lambda x: x < 10)),  # Calm wind
        Fact(uv_index=P(lambda x: x >= 5 and x <= 8)),  # Moderate UV
        Fact(weather_symbol=P(lambda x: x == 1))  # Clear sky
    )
    def recommend_beach(self):
        self.declare(Fact(activity="Beach"))
    
    # Rule for recommending "Hiking" based on moderate temperature, low winds, and low to moderate UV
    @Rule(
        Fact(t_max=P(lambda x: x > 15 and x <= 25)),  # Moderate temperature
        Fact(wind_speed=P(lambda x: x < 15)),  # Low wind
        Fact(uv_index=P(lambda x: x >= 3 and x <= 6)),  # Low to moderate UV
        Fact(weather_symbol=P(lambda x: x in [2, 3]))  # Partly cloudy or light clouds
    )
    def recommend_hiking(self):
        self.declare(Fact(activity="Hiking"))
    
    # Rule for recommending "Indoor Activities" based on strong winds or high UV
    @Rule(
        Fact(wind_speed=P(lambda x: x > 20)),  # Strong winds
        Fact(uv_index=P(lambda x: x > 8)),  # High UV
        Fact(weather_symbol=P(lambda x: x in [4, 5]))  # Cloudy or rainy
    )
    def recommend_indoor(self):
        self.declare(Fact(activity="Indoor Activities"))
    
    # Rule for recommending "Skiing" based on cold weather and moderate wind
    @Rule(
        Fact(t_max=P(lambda x: x <= 5)),  # Cold weather
        Fact(wind_speed=P(lambda x: x < 15)),  # Moderate wind
        Fact(weather_symbol=P(lambda x: x == 7))  # Snow
    )
    def recommend_skiing(self):
        self.declare(Fact(activity="Skiing"))
    
    # Rule for recommending "Cycling" based on warm weather and moderate winds
    @Rule(
        Fact(t_max=P(lambda x: x > 20 and x <= 30)),  # Warm weather
        Fact(wind_speed=P(lambda x: x < 15)),  # Moderate wind
        Fact(uv_index=P(lambda x: x >= 3 and x <= 6)),  # Moderate UV
        Fact(weather_symbol=P(lambda x: x in [2, 3]))  # Partly cloudy or light clouds
    )
    def recommend_cycling(self):
        self.declare(Fact(activity="Cycling"))
    
    # Rule for recommending "Running" based on mild weather, calm winds, and clear sky
    @Rule(
        Fact(t_max=P(lambda x: x > 10 and x <= 20)),  # Mild weather
        Fact(wind_speed=P(lambda x: x < 10)),  # Calm winds
        Fact(uv_index=P(lambda x: x >= 2 and x <= 5)),  # Moderate UV
        Fact(weather_symbol=P(lambda x: x == 1))  # Clear sky
    )
    def recommend_running(self):
        self.declare(Fact(activity="Running"))
    
    # Rule for recommending "Raincoat Shopping" if it's raining
    @Rule(
        Fact(weather_symbol=P(lambda x: x == 5)),  # Rain
    )
    def recommend_raincoat_shopping(self):
        self.declare(Fact(activity="Raincoat Shopping"))
    
    # Rule for recommending "Stay Home" based on thunderstorms or extreme weather
    @Rule(
        Fact(weather_symbol=P(lambda x: x in [14, 15])),  # Thunderstorm or drizzle
    )
    def recommend_stay_home(self):
        self.declare(Fact(activity="Stay Home"))
    
    # Rule for recommending "Picnic" based on warm, calm weather and a clear or partly cloudy sky
    @Rule(
        Fact(t_max=P(lambda x: x > 18 and x <= 28)),  # Warm weather
        Fact(wind_speed=P(lambda x: x < 10)),  # Calm winds
        Fact(uv_index=P(lambda x: x >= 3 and x <= 7)),  # Moderate UV
        Fact(weather_symbol=P(lambda x: x in [1, 2]))  # Clear or light clouds
    )
    def recommend_picnic(self):
        self.declare(Fact(activity="Picnic"))
    
    # Rule for recommending "Swimming" if hot weather, calm winds, and clear sky
    @Rule(
        Fact(t_max=P(lambda x: x > 25)),  # Hot weather
        Fact(wind_speed=P(lambda x: x < 10)),  # Calm wind
        Fact(uv_index=P(lambda x: x >= 7)),  # High UV
        Fact(weather_symbol=P(lambda x: x == 1))  # Clear sky
    )
    def recommend_swimming(self):
        self.declare(Fact(activity="Swimming"))

