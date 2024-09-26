import os
import random
# from test_code.rules3 import WeatherRecommender
from experta import *

def get_random_activity():
    try:
        script_dir = os.path.dirname(__file__)  # Get the directory of the current script
        file_path = os.path.join(script_dir, 'activities.txt')  # Create the full path
        # print(f"Looking for activities.txt at: {file_path}")  # Debug print

        with open(file_path, 'r') as file:
            activities = file.readlines()
            return random.choice(activities).strip()
    except FileNotFoundError:
        return "No activities found."


# Function to run the rule-based recommender system
def rec_activity(t_max):
# def rec_activity(t_max, wind_gust, uv_idx):
    engine = WeatherRecommender()
    engine.reset()  # Prepare the engine for a new set of facts
    
    # Provide the facts to the engine
    # engine.declare(Fact(t_max=t_max, wind_gust=wind_gust, uv_idx=uv_idx))
    engine.declare(Fact(t_max=t_max))

    # Run the engine to infer recommendations
    engine.run()
    
    # Collect the activities from the facts
    activities = [fact['activity'] for fact in engine.facts.values() if 'activity' in fact]
    
    # If no rule-based activity is found, get a random activity from activities.txt
    if not activities:
        random_activity = get_random_activity()
        return [random_activity]  # Return as a list to match format
    else:
        return activities
