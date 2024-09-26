import os
import random
import ast
# from test_code.rules3 import WeatherRecommender
from experta import *

# Function to get random activity from static2.txt
def get_random_static():
    try:
        # Get the full path to static2.txt
        script_dir = os.path.dirname(__file__)  # Get the directory of the current script
        file_path = os.path.join(script_dir, 'static2.txt')  # Create the full path

        # Open and read the static2.txt file
        activity = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            activities = [ast.literal_eval(line.strip()) for line in lines]  # Parse the activity list from each line
            
            # Randomly select one activity
            selected_activity = random.choice(activities)
            
            # Extract the details of the selected activity
            activity_name = selected_activity[0]
            description = selected_activity[1]
            min_temp = selected_activity[2]
            max_temp = selected_activity[3]
            uv_index = selected_activity[4]
            
            # Print the randomly selected activity
            print(f"\n{activity_name}")
            print(f"{description}\n")
            print(f"Temperature Range: {min_temp}°C - {max_temp}°C")
            print(f"UV Index: {uv_index}")
            activity = [activity_name,description,min_temp,max_temp,uv_index]
    except FileNotFoundError:
        print("Error: 'static2.txt' file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return activity


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
