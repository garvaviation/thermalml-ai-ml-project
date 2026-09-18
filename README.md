# ThermalML — Spacecraft Component Temperature Prediction & Overheat Anomaly Detection

## Overview

ThermalML is a machine learning project (built for the *Fundamentals of AI and ML* course)
that predicts the **steady-state temperature of a spacecraft component in low Earth orbit**
and flags whether it is at risk of **overheating**, using orbital and thermal-design
parameters as input.

Because real spacecraft telemetry is proprietary and not publicly available, the project
generates a **physics-based synthetic dataset** using the same radiative energy balance
equation used in real spacecraft thermal design (`Q_absorbed = Q_emitted`, solved for
steady-state temperature). This keeps the data scientifically grounded while making the
project fully reproducible.

## Features

- **Synthetic data generation** from a first-order orbital thermal-balance model (solar,
  albedo, and Earth-IR flux; eclipse fraction; surface absorptivity/emissivity).
- **Regression module** — predicts component temperature (°C) using Linear Regression and
  Random Forest, automatically selecting the best-performing model.
- **Classification module** — flags "overheat risk" using a Random Forest classifier.
- **Evaluation & visualization module** — RMSE/MAE/R² for regression, Accuracy/Precision/
  Recall/F1 + confusion matrix for classification, feature-importance plots.
- **CLI interface** to generate data, train, evaluate, and run predictions.
- **Unit tests** covering data generation and preprocessing/validation logic.
- **Logging** throughout the pipeline for traceability.

## Technologies Used

- Python 3.10+
- pandas, NumPy — data handling & synthesis
- scikit-learn — regression, classification, preprocessing, metrics
- Matplotlib — visualization
- joblib — model persistence
- unittest — testing

## Project Structure

```
thermal-ml-project/
├── main.py                    # CLI entry point (orchestrates the full pipeline)
├── requirements.txt
├── README.md
├── statement.md
├── src/
│   ├── config.py               # central configuration/constants
│   ├── logger.py                # logging utility
│   ├── data_generator.py        # Module 1a: synthetic dataset generation
│   ├── preprocessing.py         # Module 1b: validation, cleaning, scaling
│   ├── train_regression.py      # Module 2a: temperature regression
│   ├── train_classifier.py      # Module 2b: overheat classification
│   ├── predict.py               # prediction engine (loads saved models)
│   └── visualize.py             # Module 3: evaluation plots
├── tests/
│   └── test_pipeline.py         # unit tests
├── data/                        # generated dataset (thermal_dataset.csv)
├── models/                      # persisted trained models (.joblib)
└── docs/                        # diagrams + result plots (used in the report)
```

## Setup & Installation

### 1. Prerequisites
- Python 3.10 or later installed (`python3 --version` to check)
- `pip` package manager
- Git installed

### 2. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 3. Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Project

Run the entire pipeline (generate data → train → evaluate → sample prediction) in one command:
```bash
python main.py --all
```

Or run each stage individually:
```bash
python main.py --generate-data   # regenerate the synthetic dataset
python main.py --train           # train + evaluate regression and classification models
python main.py --predict         # run a single sample prediction
```

Generated outputs:
- `data/thermal_dataset.csv` — the synthetic dataset
- `models/*.joblib` — trained regression model, classifier, and scaler
- `docs/*.png` — evaluation and diagram plots

### Using the predictor in your own code
```python
from src.predict import ThermalPredictor

predictor = ThermalPredictor()
result = predictor.predict({
    "altitude_km": 550,
    "sun_angle_deg": 15,
    "absorptivity": 0.6,
    "emissivity": 0.8,
    "surface_area_m2": 0.5,
    "eclipse_fraction": 0.1,
    "albedo_coefficient": 0.3,
    "solar_flux_wm2": 1300,
    "albedo_flux_wm2": 180,
    "earth_ir_flux_wm2": 230,
})
print(result)
# {'predicted_temp_c': 107.93, 'overheat_risk': True, 'overheat_probability': 0.8718}
```

## Testing

Run the unit test suite:
```bash
python -m unittest discover tests -v
```

All 6 tests should pass, covering dataset generation, physical plausibility of generated
temperatures, and preprocessing/validation error handling.

## Results

| Model | RMSE (°C) | MAE (°C) | R² |
|---|---|---|---|
| Linear Regression | 13.55 | 10.71 | 0.9499 |
| Random Forest (selected) | 7.54 | 5.91 | **0.9845** |

| Classifier metric | Value |
|---|---|
| Accuracy | 0.965 |
| Precision | 0.960 |
| Recall | 0.958 |
| F1-score | 0.959 |

See `docs/` for the full set of plots (temperature distribution, predicted-vs-actual,
confusion matrix, feature importance, architecture/workflow/UML diagrams).

## Screenshots

See the `docs/` folder:
- `architecture_diagram.png` — system architecture
- `workflow_diagram.png` — process workflow
- `uml_class_diagram.png` / `sequence_diagram.png` — UML diagrams
- `predicted_vs_actual.png`, `confusion_matrix.png`, `feature_importance_*.png` — model results

## Author

Built as a course project for *Fundamentals of AI and ML*, applying the author's
background in space systems thermal engineering.
