import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter

class TrackCharacteristics:
    def __init__(self, name, length, avg_speed, turns, high_speed_corners, med_speed_corners, low_speed_corners):
        self.name = name
        self.length = length  # km
        self.avg_speed = avg_speed  # km/h
        self.turns = turns
        self.high_speed_corners = high_speed_corners
        self.med_speed_corners = med_speed_corners
        self.low_speed_corners = low_speed_corners
        
        # Calculate derived characteristics
        self.corner_ratio = (high_speed_corners + med_speed_corners + low_speed_corners) / turns
        self.high_speed_ratio = high_speed_corners / turns
        self.baseline_downforce = self._calculate_baseline_downforce()
        
    def _calculate_baseline_downforce(self):
        # Higher value means more downforce needed
        # Influenced by corner speeds and total turns
        return (self.low_speed_corners * 1.5 + self.med_speed_corners * 1.0 + 
                self.high_speed_corners * 0.5) / self.turns

# Track database
TRACK_DATABASE = {
    "Silverstone": TrackCharacteristics(
        name="Silverstone",
        length=5.891,
        avg_speed=237,
        turns=18,
        high_speed_corners=7,
        med_speed_corners=6,
        low_speed_corners=5
    ),
    "Monza": TrackCharacteristics(
        name="Monza",
        length=5.793,
        avg_speed=264,
        turns=11,
        high_speed_corners=7,
        med_speed_corners=2,
        low_speed_corners=2
    ),
    "Monaco": TrackCharacteristics(
        name="Monaco",
        length=3.337,
        avg_speed=157,
        turns=19,
        high_speed_corners=2,
        med_speed_corners=5,
        low_speed_corners=12
    ),
    "Spa": TrackCharacteristics(
        name="Spa",
        length=7.004,
        avg_speed=238,
        turns=19,
        high_speed_corners=9,
        med_speed_corners=5,
        low_speed_corners=5
    )
}

feedback_categories = [
            'overall_balance',
            'front_grip', 
            'rear_grip',
            'straight_line_stability',
            'corner_entry',
            'corner_exit',
            'brake_stability',
            'traction',
            'kerb_riding',
            'turn_in_response',
            'mid_corner_balance',
            'tire_degradation'
        ]

def generate_sample_data(track_name="Silverstone", seed=None):
    """
    Generate sample F1 track and weather data for the aero optimization pipeline.
    """
    if seed is None:
        seed = int(time.time() * 1000) % 2**32
    np.random.seed(seed)
    
    # Get track characteristics
    track = TRACK_DATABASE[track_name]
    
    # Number of samples (laps) - vary slightly but consider track length
    # Longer tracks typically have fewer laps in a session
    base_samples = int(400 / track.length)  # Approximate session distance of 400km
    n_samples = np.random.randint(base_samples - 5, base_samples + 5)
    
    def generate_speed_profile(length_km, turns, high_speed_ratio):
        """Generate track-specific speed profile"""
        points_per_km = 100 / length_km
        num_points = 100  # We'll keep 100 points but vary the pattern
        
        # Create base profile considering track characteristics
        x = np.linspace(0, 2*np.pi, num_points)
        
        # More high-frequency components for tracks with more corners
        profile = np.zeros(num_points)
        for i in range(1, turns//2):
            amplitude = 1.0 / (i * (1 + (1-high_speed_ratio)))
            profile += amplitude * np.sin(i * x)
        
        # Normalize and scale to appropriate speed range
        profile = (profile - profile.min()) / (profile.max() - profile.min())
        min_speed = 80 if track.name == "Monaco" else 120
        speed_range = track.avg_speed * 1.3 - min_speed
        return profile * speed_range + min_speed

    def generate_cfd_data():
        """Generate CFD simulation data with track-specific characteristics"""
        cfd_data = np.zeros((n_samples, 32, 32, 16))
        
        # Base pressure and velocity distributions affected by track characteristics
        base_pressure = 1.0 + track.baseline_downforce * 0.1
        base_velocity = track.avg_speed / 100  # Normalize to reasonable range
        
        for i in range(n_samples):
            # Pressure distribution (channels 0-7)
            cfd_data[i, :, :, 0:8] = np.random.normal(0, 1, (32, 32, 8)) * 0.1 + base_pressure
            
            # Velocity fields (channels 8-15)
            cfd_data[i, :, :, 8:16] = np.random.normal(0, 1, (32, 32, 8)) * 0.2 + base_velocity
            
            for c in range(16):
                cfd_data[i, :, :, c] = gaussian_filter(cfd_data[i, :, :, c], sigma=1.0)
            
            # Evolution over session
            base_pressure += np.random.normal(0, 0.001)
            base_velocity += np.random.normal(0, 0.002)
        
        return cfd_data

    def generate_telemetry_data():
        """Generate car telemetry data with track-specific characteristics"""
        telemetry = np.zeros((n_samples, 100, 32))
        
        # Generate speed profile based on track characteristics
        speed_profile = generate_speed_profile(track.length, track.turns, track.high_speed_ratio)
        
        for i in range(n_samples):
            # Speed with lap-by-lap variation
            telemetry[i, :, 0] = speed_profile + np.random.normal(0, track.avg_speed * 0.02, 100)
            
            # Downforce varies with track characteristics
            downforce_coef = 0.0015 * (1 + track.baseline_downforce * 0.1)
            telemetry[i, :, 1] = (telemetry[i, :, 0] ** 2) * downforce_coef + np.random.normal(0, 50, 100)
            
            # Brake temperatures - higher for tracks with more low speed corners
            base_brake_temp = 380 + track.low_speed_corners * 2
            for wheel in range(4):
                telemetry[i, :, 2+wheel] = base_brake_temp + np.random.normal(0, 20, 100)
            
            # Tire temperatures
            base_tire_temp = 80 + track.corner_ratio * 5 + i * 0.1
            for wheel in range(4):
                telemetry[i, :, 6+wheel] = base_tire_temp + np.random.normal(0, 5, 100)
            
            # Engine parameters adjusted for track
            rpm_profile = generate_speed_profile(track.length, track.turns, track.high_speed_ratio) * 50 + 8000
            telemetry[i, :, 10] = rpm_profile + np.random.normal(0, 200, 100)
            telemetry[i, :, 11] = 95 + track.avg_speed/100 + np.random.normal(0, 2, 100)
            telemetry[i, :, 12] = 3 + np.random.normal(0, 0.2, 100)
            
            # Aerodynamic sensors
            for sensor in range(19):
                phase_shift = np.random.uniform(0, np.pi)
                base_signal = np.sin(np.linspace(0, 4*np.pi + phase_shift, 100)) * 0.5
                noise = np.random.normal(0, 0.2, 100)
                telemetry[i, :, 13+sensor] = base_signal + noise
                telemetry[i, :, 13+sensor] = gaussian_filter(telemetry[i, :, 13+sensor], sigma=1.0)
        
        return telemetry

    def generate_weather_data():
        """Generate weather forecast data considering track location"""
        weather_data = np.zeros((n_samples, 8))
        
        # Base conditions vary by track
        if track_name == "Monaco":
            base_temp = np.random.normal(22, 2)  # Mediterranean climate
            base_humidity = np.random.normal(70, 5)  # Coastal humidity
        elif track_name == "Silverstone":
            base_temp = np.random.normal(18, 3)  # UK climate
            base_humidity = np.random.normal(75, 5)  # Usually humid
        elif track_name == "Monza":
            base_temp = np.random.normal(25, 2)  # Italian climate
            base_humidity = np.random.normal(65, 5)
        else:  # Spa
            base_temp = np.random.normal(20, 3)  # Variable climate
            base_humidity = np.random.normal(80, 5)  # Often wet
        
        base_pressure = np.random.normal(1013, 2)
        
        for i in range(n_samples):
            # Temperature evolution
            weather_data[i, 0] = base_temp + np.random.normal(0, 0.5)
            base_temp += np.random.normal(0, 0.1)
            
            # Humidity with temperature correlation
            weather_data[i, 1] = base_humidity - (weather_data[i, 0] - 20) * 2 + np.random.normal(0, 2)
            
            # Wind conditions - stronger at exposed tracks
            wind_factor = 1.5 if track_name in ["Silverstone", "Spa"] else 1.0
            weather_data[i, 2] = np.random.exponential(8) * wind_factor
            weather_data[i, 3] = (i/n_samples * 360 + np.random.normal(0, 10)) % 360
            
            weather_data[i, 4] = base_pressure + np.random.normal(0, 0.5)
            weather_data[i, 5] = weather_data[i, 0] + np.random.normal(15, 2)
            
            # Cloud cover and rain chance - track specific
            if track_name in ["Spa", "Silverstone"]:
                base_cloud = 60  # More likely to be cloudy
                rain_factor = 1.2
            else:
                base_cloud = 40
                rain_factor = 0.8
            
            cloud_cover = np.clip(np.random.normal(base_cloud, 15), 0, 100)
            weather_data[i, 6] = cloud_cover
            weather_data[i, 7] = np.clip(cloud_cover * 0.5 * rain_factor + np.random.normal(0, 10), 0, 100)
        
        # Apply temporal smoothing
        for j in range(weather_data.shape[1]):
            weather_data[:, j] = gaussian_filter(weather_data[:, j], sigma=2.0)
        
        return weather_data

    def generate_driver_feedback():
        """Generate structured driver feedback data with track-specific characteristics"""
        
        feedback_data = np.zeros((n_samples, len(feedback_categories)))
        feedback_dict_list = []  # To store feedback with category labels
        
        # Base satisfaction levels that evolve over the session
        base_satisfaction = np.random.normal(7, 0.5)
        
        # Track-specific characteristics influence feedback
        track_factors = {
            'straight_line_stability': 1.0 + (track.high_speed_corners / track.turns) * 0.2,
            'corner_entry': 1.0 - (track.low_speed_corners / track.turns) * 0.1,
            'corner_exit': 1.0 - (track.low_speed_corners / track.turns) * 0.1,
            'brake_stability': 1.0 - (track.low_speed_corners / track.turns) * 0.15,
            'kerb_riding': 1.0 + (track.med_speed_corners / track.turns) * 0.1,
            'tire_degradation': 1.0 + (track.high_speed_corners / track.turns) * 0.2
        }
        
        for i in range(n_samples):
            feedback_dict = {}
            
            # Overall balance affects other parameters
            overall_balance = base_satisfaction + np.random.normal(0, 0.3)
            feedback_dict['overall_balance'] = overall_balance
            
            # Grip levels with correlation to overall balance
            front_grip = overall_balance * (1 + np.random.normal(0, 0.1))
            rear_grip = overall_balance * (1 + np.random.normal(0, 0.1))
            feedback_dict['front_grip'] = front_grip
            feedback_dict['rear_grip'] = rear_grip
            
            # Track-specific feedback
            for category in feedback_categories[3:]:  # Skip first 3 which we handled above
                base_value = 7.0
                if category in track_factors:
                    base_value *= track_factors[category]
                
                # Add correlation with overall balance
                value = base_value + (overall_balance - 7.0) * 0.3 + np.random.normal(0, 0.5)
                feedback_dict[category] = value
            
            # Evolution over session
            base_satisfaction += np.random.normal(0, 0.1)
            
            # Special handling for tire degradation
            feedback_dict['tire_degradation'] -= (i / n_samples) * 2.0  # Progressive degradation
            
            # Store in array and dictionary list
            for j, category in enumerate(feedback_categories):
                feedback_data[i, j] = feedback_dict[category]
            feedback_dict_list.append(feedback_dict)
            
            # Clip values to valid range (1-10)
            feedback_data[i] = np.clip(feedback_data[i], 1, 10)
            for key in feedback_dict:
                feedback_dict[key] = np.clip(feedback_dict[key], 1, 10)
        
        return feedback_data, feedback_dict_list
    
    def generate_initial_setup():
        """Generate track-specific initial setup"""
        downforce_level = track.baseline_downforce
        
        setup = {
            'front_wing_angle': 28.5 + downforce_level * 2 + np.random.normal(0, 0.5),
            'rear_wing_angle': 24.0 + downforce_level * 2 + np.random.normal(0, 0.5),
            'brake_balance': 57.5 + (track.low_speed_corners - track.high_speed_corners) * 0.2 + np.random.normal(0, 0.3),
            'diff_entry': 65.0 + np.random.normal(0, 1.0),
            'diff_mid': 60.0 + np.random.normal(0, 1.0),
            'diff_exit': 75.0 + np.random.normal(0, 1.0),
            'ride_height_front': 25.0 + (track.high_speed_corners * 0.1) + np.random.normal(0, 0.2),
            'ride_height_rear': 45.0 + (track.high_speed_corners * 0.1) + np.random.normal(0, 0.2),
            'anti_roll_bar_front': int(np.clip(7 + (track.high_speed_corners - track.low_speed_corners) * 0.2 + np.random.randint(-1, 2), 1, 11)),
            'anti_roll_bar_rear': int(np.clip(6 + (track.high_speed_corners - track.low_speed_corners) * 0.2 + np.random.randint(-1, 2), 1, 11))
        }
        
        return setup

    # Generate all components
    feedback_data, feedback_dict_list = generate_driver_feedback()
    track_data = {
        'cfd': generate_cfd_data(),
        'telemetry': generate_telemetry_data(),
        'feedback': feedback_data,
        'feedback_categories': feedback_categories,
        'feedback_dict_list': feedback_dict_list
    }
    
    weather_forecast = generate_weather_data()
    initial_setup = generate_initial_setup()
    
    return track_data, weather_forecast, initial_setup

    