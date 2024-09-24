import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from data_generation import generate_sample_data, TRACK_DATABASE
# Custom dataset for handling multiple data sources
class F1AeroDataset(Dataset):
    def __init__(self, cfd_data, telemetry_data, weather_data, driver_feedback):
        # Transpose CFD data to PyTorch format (batch_size, channels, height, width)
        self.cfd_data = torch.FloatTensor(np.transpose(cfd_data, (0, 3, 1, 2)))
        self.telemetry = torch.FloatTensor(telemetry_data)
        self.weather = torch.FloatTensor(weather_data)
        self.driver_feedback = torch.FloatTensor(driver_feedback)
        
    def __len__(self):
        return len(self.cfd_data)
    
    def __getitem__(self, idx):
        return {
            'cfd': self.cfd_data[idx],
            'telemetry': self.telemetry[idx],
            'weather': self.weather[idx],
            'feedback': self.driver_feedback[idx]
        }

# CNN module for processing CFD data
class CFDProcessor(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, hidden_dim)  # Adjusted for 32x32 input
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(self.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = x.view(x.size(0), -1)  # Flatten: 64 * 8 * 8
        return self.fc(x)

# LSTM module for processing temporal data
class TemporalProcessor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Take only the last time step

# Main model combining all components
class AeroOptimizer(pl.LightningModule):
    def __init__(self, cfd_channels, telemetry_features, weather_features, feedback_features):
        super().__init__()
        self.hidden_dim = 128
        
        # Component processors
        self.cfd_processor = CFDProcessor(cfd_channels, self.hidden_dim)
        self.telemetry_processor = TemporalProcessor(telemetry_features, self.hidden_dim)
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + weather_features + feedback_features, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, 64)
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
    def forward(self, batch):
        # Process CFD data
        cfd_features = self.cfd_processor(batch['cfd'])
        
        # Process telemetry data
        telemetry_features = self.telemetry_processor(batch['telemetry'])
        
        # Concatenate all features
        combined = torch.cat([
            cfd_features,
            telemetry_features,
            batch['weather'],
            batch['feedback']
        ], dim=1)
        
        return self.fusion_layer(combined)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        # Assuming target is part of batch
        loss = self.loss_fn(outputs, batch['target'])
        self.log('train_loss', loss)
        return loss

# XGBoost final predictor
class SetupOptimizer:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8
        )
        self.scaler = StandardScaler()
        
    def train(self, deep_features, setup_params, performance_metrics):
        # Repeat setup parameters for each sample
        n_samples = deep_features.shape[0]
        setup_params_repeated = np.tile(setup_params, (n_samples, 1))
        
        # Combine deep features with setup parameters
        X = np.concatenate([deep_features, setup_params_repeated], axis=1)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train XGBoost model
        self.model.fit(X_scaled, performance_metrics)
        
    def predict(self, deep_features, setup_params):
        # Repeat setup parameters for each sample
        n_samples = deep_features.shape[0]
        setup_params_repeated = np.tile(setup_params, (n_samples, 1))
        
        X = np.concatenate([deep_features, setup_params_repeated], axis=1)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def optimize_aero_setup(track_data, weather_forecast, initial_setup):
    # Convert initial_setup dict to numpy array
    setup_array = np.array([list(initial_setup.values())])
    
    # Initialize models
    deep_model = AeroOptimizer(
        cfd_channels=16,
        telemetry_features=32,
        weather_features=8,
        feedback_features=12
    )
    
    # Create dataset
    dataset = F1AeroDataset(
        track_data['cfd'],
        track_data['telemetry'],
        weather_forecast,
        track_data['feedback']
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    
    # Get deep features
    batch = next(iter(dataloader))
    with torch.no_grad():
        deep_features = deep_model(batch).numpy()
    
    # Initialize setup optimizer
    setup_optimizer = SetupOptimizer()
    
    # Create synthetic performance metrics (lap times)
    # In reality, these would come from simulation or real-world data
    n_samples = len(dataset)
    base_laptime = 80.0  # Base laptime in seconds
    performance_metrics = base_laptime + np.random.normal(0, 0.5, n_samples)
    
    # Train the setup optimizer
    setup_optimizer.train(deep_features, setup_array, performance_metrics)
    
    # Get optimized setup
    optimal_setup_values = setup_optimizer.predict(deep_features, setup_array)
    
    # Average the predictions to get final setup
    optimal_setup_values = optimal_setup_values.mean()
    
    # Convert back to dictionary
    optimal_setup = dict(zip(initial_setup.keys(), [optimal_setup_values] * len(initial_setup)))
    
    # Apply realistic constraints to the optimized values
    optimal_setup = constrain_setup_parameters(optimal_setup)
    
    return optimal_setup

def constrain_setup_parameters(setup):
    """Apply realistic constraints to setup parameters"""
    constraints = {
        'front_wing_angle': (20.0, 35.0),
        'rear_wing_angle': (15.0, 30.0),
        'brake_balance': (50.0, 65.0),
        'diff_entry': (50.0, 80.0),
        'diff_mid': (50.0, 75.0),
        'diff_exit': (60.0, 85.0),
        'ride_height_front': (15.0, 35.0),
        'ride_height_rear': (35.0, 55.0),
        'anti_roll_bar_front': (1, 11),
        'anti_roll_bar_rear': (1, 11)
    }
    
    constrained_setup = {}
    for param, value in setup.items():
        min_val, max_val = constraints[param]
        constrained_setup[param] = np.clip(value, min_val, max_val)
    
    return constrained_setup

# Example of how to use the data with our optimization pipeline
def run_optimization_example():
    # Generate the data
    tracks = ["Monaco", "Monza", "Silverstone", "Spa"]
    
    for track_name in tracks:
        print(f"\n=== {track_name} ===")
        track_data, weather, setup = generate_sample_data(track_name)
        
        print(f"Track Characteristics:")
        track = TRACK_DATABASE[track_name]
        print(f"Length: {track.length}km")
        print(f"Average Speed: {track.avg_speed}km/h")
        print(f"Turns: {track.turns} (High: {track.high_speed_corners}, Med: {track.med_speed_corners}, Low: {track.low_speed_corners})")
        
        print("\nInitial Setup:")
        for param, value in setup.items():
            print(f"{param}: {value:.2f}")
        
        print(f"\nWeather Sample:")
        print(f"Temperature: {weather[0, 0]:.1f}°C")
        print(f"Rain Chance: {weather[0, 7]:.1f}%")
        
        print(f"\nDriver Feedback (First Lap):")
        for category, value in zip(track_data['feedback_categories'], track_data['feedback'][0]):
            print(f"{category}: {value:.1f}")
        
        # Show feedback evolution
        print(f"\nFeedback Evolution (Tire Degradation):")
        lap_samples = [0, len(track_data['feedback'])//2, -1]  # Start, middle, end
        for lap in lap_samples:
            print(f"Lap {lap}: {track_data['feedback'][lap, track_data['feedback_categories'].index('tire_degradation')]:.1f}")

if __name__ == "__main__":
    optimal_setup = run_optimization_example()