import torch
import pickle
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
#from utils.predict_model import CNNWeatherPredictor
#import torch
#import torch.nn as nn

def load_scaler(scaler_path):
    """
    Load the scaler from the pickle file.
    """
    file = open(scaler_path, 'rb')
    scaler = pickle.load(file)

    return scaler
def scale_data(scaler,df):
    """
    Load the scaler from the pickle file.
    """
    df_np = df.to_numpy()
    df_scaler = scaler.transform(df_np)
    data_scaled = torch.tensor(df_scaler, dtype=torch.float32).unsqueeze(0)

    return data_scaled

def load_model(model_path):
    """
    Load the pre-trained model.
    """
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    return model.eval()

def make_prediction(model,data_scaled,scaler):
    with torch.no_grad():  # Disable gradients for inference
        prediction = model(data_scaled)
    #print(prediction)
        # Inverse transform the predictions to get them back to the original scale
    predicted_values = prediction.cpu().numpy().reshape(-1, 6)  # Convert to numpy
    predicted_values = scaler.inverse_transform(predicted_values)
    print('====')
    print(predicted_values)
    return predicted_values

