import re
from datetime import datetime, timedelta

def preprocess_relative_dates(user_input):
    user_input_lower = user_input.lower()
    
    past_match = re.search(r'(\d+) (days?|weeks?|months?) ago', user_input_lower)
    if past_match:
        return None, "Sorry, I can only predict the weather for future dates."

    if 'today' in user_input_lower:
        return "today", "today"
    if 'tomorrow' in user_input_lower:
        return "tomorrow", "tomorrow"

    future_match = re.search(r'in (\d+) (days?|weeks?|months?)', user_input_lower)
    if future_match:
        return future_match.group(1) + " " + future_match.group(2), None

    weekday_match = re.search(r'next (monday|tuesday|wednesday)', user_input_lower)
    if weekday_match:
        weekday = weekday_match.group(1)
        return get_next_weekday(weekday), "next " + weekday
    
    return user_input, None

def get_next_weekday(weekday_name):
    days_of_week = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3}
    today = datetime.now()
    current_weekday = today.weekday()
    target_weekday = days_of_week.get(weekday_name.lower())
    days_until_next_weekday = (target_weekday - current_weekday + 7) % 7
    if days_until_next_weekday == 0:
        days_until_next_weekday = 7
    next_weekday = today + timedelta(days=days_until_next_weekday)
    return next_weekday


def find_australian_city_in_input(user_input):
    # Predefined list of Australian cities
    australian_cities = [
        "Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Canberra",
        "Hobart", "Darwin", "Gold Coast", "Newcastle", "Wollongong", 
        "Geelong", "Townsville", "Cairns", "Toowoomba", "Ballarat"
    ]
    for city in australian_cities:
        if re.search(r'\b' + re.escape(city) + r'\b', user_input, re.IGNORECASE):
            return city
    return None

def is_weather_related(user_input):
    # Define weather-related keywords
    weather_keywords = ['weather', 'temperature', 'forecast', 'rain', 'sun', 'humidity', 'wind']
   
    # Check if any weather-related keywords are in the user input
    for keyword in weather_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', user_input, re.IGNORECASE):
            return True
    return False

def preprocess_relative_dates(user_input):
    user_input_lower = user_input.lower()
    
    # Handle past phrases like "X days ago", "X weeks ago", "X months ago"
    past_match = re.search(r'(\d+) (days?|weeks?|months?) ago', user_input_lower)
    if past_match:
        return None, "Sorry, I can only predict the weather for future dates."  # Warn the user if a past date is detected

    # Handle specific cases like "today" or "tomorrow" and return them directly
    if 'today' in user_input_lower:
        return "today", "today"
    if 'tomorrow' in user_input_lower:
        return "tomorrow", "tomorrow"

    # Handle future phrases like "in X days", "in X weeks", "in X months"
    future_match = re.search(r'in (\d+) (days?|weeks?|months?)', user_input_lower)
    if future_match:
        number = int(future_match.group(1))
        unit = future_match.group(2)
        return future_match.group(1) + " " + future_match.group(2),None

        # Handle "next week" or "next month"
    if 'next week' in user_input_lower:
        return "next week", "next week"
    if 'next month' in user_input_lower:
        return "next month", "next month"

    # Handle specific weekdays like "next Monday"
    weekday_match = re.search(r'next (monday|tuesday|wednesday|thursday|friday|saturday|sunday)', user_input_lower)
    if weekday_match:
        weekday = weekday_match.group(1)
        next_weekday = get_next_weekday(weekday)
        if next_weekday:
            return next_weekday.strftime('%Y-%m-%d'), "next" + " " + weekday
            #return next_weekday.strftime('%Y-%m-%d'), None
    
    return user_input, None