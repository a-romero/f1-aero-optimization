# F1 Aerodynamic Configuration Optimization

A comprehensive machine learning system for optimizing Formula 1 car aerodynamic configurations using CFD data, telemetry, and driver feedback. The system combines deep learning techniques with gradient boosting to provide optimal setup recommendations for different tracks and conditions.

## Overview

This project implements a comprehensive pipeline for F1 car setup optimization, focusing on aerodynamic configuration. It processes multiple data sources:
- CFD (Computational Fluid Dynamics) simulations
- Real-time telemetry data
- Track-specific characteristics
- Weather conditions
- Structured driver feedback

## Architecture

### Deep Learning Pipeline
- **CNN Module**: Processes spatial CFD data to extract flow patterns
- **LSTM Module**: Handles temporal telemetry data sequences
- **Fusion Layer**: Combines features from multiple data sources
- **XGBoost Optimizer**: Final setup parameter optimization

### Data Processing
- CFD Data: 32x32 spatial grid with 16 channels (pressure and velocity fields)
- Telemetry: 32 sensors including speed, temperatures, and aerodynamic measurements
- Weather: 8 parameters including temperature, humidity, wind conditions
- Driver Feedback: 12 structured metrics for car handling characteristics

## Installation

```bash
# Clone the repository
git clone https://github.com/a-romero/f1-aero-optimization.git
cd f1-aero-optimization

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- SciPy
- XGBoost
- PyTorch Lightning

## Usage

### Basic Usage

```python
from aero_optimization import optimize_aero_setup
from data_generation import generate_sample_data

# Generate sample data for a specific track
track_data, weather_forecast, initial_setup = generate_sample_data("Silverstone")

# Run optimization
optimal_setup = optimize_aero_setup(track_data, weather_forecast, initial_setup)

# Print optimized setup
for param, value in optimal_setup.items():
    print(f"{param}: {value:.2f}")
```

### Track-Specific Configuration

The system supports different F1 circuits, each with its unique characteristics:
```python
tracks = ["Monaco", "Monza", "Silverstone", "Spa"]
for track in tracks:
    track_data, weather, setup = generate_sample_data(track)
    optimal_setup = optimize_aero_setup(track_data, weather, setup)
```

## Data Structure

### CFD Data
```python
cfd_data.shape = (n_samples, 32, 32, 16)
# - n_samples: Number of measurements
# - 32x32: Spatial grid points
# - 16 channels: Pressure (0-7) and velocity (8-15) fields
```

### Telemetry Data
```python
telemetry_data.shape = (n_samples, 100, 32)
# - n_samples: Number of laps
# - 100: Time steps per lap
# - 32: Sensor measurements
```

### Driver Feedback Categories
- Overall balance
- Front/rear grip
- Straight-line stability
- Corner entry/exit performance
- Brake stability
- Traction
- Kerb riding
- Turn-in response
- Mid-corner balance
- Tire degradation

## Model Parameters

### Aerodynamic Setup Parameters
- Front/rear wing angles (degrees)
- Brake balance (% forward)
- Differential settings (entry/mid/exit %)
- Ride heights (mm)
- Anti-roll bar settings (1-11 scale)

## Future Improvements

- Real-time optimization during race weekends
- Integration with wind tunnel data
- Enhanced weather impact modeling
- Driver preference learning
- Multi-objective optimization for different racing scenarios

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

* **Alberto Romero** - [@a-romero](https://github.com/a-romero)