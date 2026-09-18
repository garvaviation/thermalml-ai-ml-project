"""
main.py
-------
ThermalML - Spacecraft Component Temperature Prediction & Overheat
Anomaly Detection.

CLI entry point that orchestrates the full pipeline:
  1. Generate/refresh the synthetic dataset
  2. Preprocess data
  3. Train regression + classification models
  4. Evaluate and generate plots
  5. Optionally run a single interactive prediction

Usage:
    python main.py --generate-data     # regenerate the synthetic dataset
    python main.py --train             # train and evaluate both models
    python main.py --predict           # run a sample prediction
    python main.py --all               # do everything, in order
"""

import argparse
import sys

from src.logger import get_logger
from src.data_generator import generate_dataset, save_dataset
from src.preprocessing import run_preprocessing_pipeline, split_and_scale, REGRESSION_TARGET, CLASSIFICATION_TARGET
from src.train_regression import train_models as train_reg_models, evaluate_models as eval_reg_models, select_and_save_best
from src.train_classifier import train_classifier, evaluate_classifier
from src.predict import ThermalPredictor
from src.visualize import (
    plot_temperature_distribution,
    plot_predicted_vs_actual,
    plot_confusion_matrix,
    plot_feature_importance,
)
from src.preprocessing import FEATURE_COLUMNS
import joblib
from src.config import CLASSIFIER_MODEL_PATH

logger = get_logger(__name__)


def cmd_generate_data():
    df = generate_dataset()
    save_dataset(df)
    plot_temperature_distribution(df)


def cmd_train():
    df = run_preprocessing_pipeline()

    # --- Regression ---
    X_train, X_test, y_train, y_test = split_and_scale(df, REGRESSION_TARGET, fit_scaler=True)
    reg_models = train_reg_models(X_train, y_train)
    reg_results = eval_reg_models(reg_models, X_test, y_test)
    best_name, best_model = select_and_save_best(reg_models, reg_results)
    plot_predicted_vs_actual(y_test, best_model.predict(X_test))
    import os as _os
    from src.config import DOCS_DIR as _DOCS_DIR
    plot_feature_importance(
        reg_models.get("random_forest"), FEATURE_COLUMNS,
        title="Feature Importance - Temperature Regression",
        path=_os.path.join(_DOCS_DIR, "feature_importance_regression.png"),
    )

    # --- Classification (reuse the scaler already fit above) ---
    X_train_c, X_test_c, y_train_c, y_test_c = split_and_scale(
        df, CLASSIFICATION_TARGET, fit_scaler=False
    )
    clf = train_classifier(X_train_c, y_train_c)
    clf_metrics = evaluate_classifier(clf, X_test_c, y_test_c)
    joblib.dump(clf, CLASSIFIER_MODEL_PATH)
    plot_confusion_matrix(y_test_c, clf.predict(X_test_c))
    import os as _os
    from src.config import DOCS_DIR as _DOCS_DIR
    plot_feature_importance(
        clf, FEATURE_COLUMNS,
        title="Feature Importance - Overheat Classifier",
        path=_os.path.join(_DOCS_DIR, "feature_importance_classifier.png"),
    )

    print("\n=== Regression results ===")
    for name, m in reg_results.items():
        print(f"  {name}: RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  R2={m['r2']:.4f}")
    print(f"  Best model selected: {best_name}")

    print("\n=== Classification results ===")
    for k, v in clf_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}")
    print(f"  confusion_matrix: {clf_metrics['confusion_matrix']}")


def cmd_predict():
    predictor = ThermalPredictor()
    sample = {
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
    }
    result = predictor.predict(sample)
    print("Sample input:", sample)
    print("Prediction:", result)


def main():
    parser = argparse.ArgumentParser(description="ThermalML pipeline runner")
    parser.add_argument("--generate-data", action="store_true", help="Regenerate synthetic dataset")
    parser.add_argument("--train", action="store_true", help="Train and evaluate models")
    parser.add_argument("--predict", action="store_true", help="Run a sample prediction")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline end to end")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    if args.all or args.generate_data:
        cmd_generate_data()
    if args.all or args.train:
        cmd_train()
    if args.all or args.predict:
        cmd_predict()


if __name__ == "__main__":
    main()
