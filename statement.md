# Problem Statement

## Problem Statement

Spacecraft components in low Earth orbit (LEO) experience continuously varying thermal
loads from solar radiation, Earth albedo, and Earth infrared emission, modulated by
orbital altitude, sun angle, and eclipse periods. Predicting a component's steady-state
temperature — and identifying when it risks exceeding a safe operating threshold — is a
core task in spacecraft thermal design, traditionally done with finite-difference
thermal models. This project explores whether machine learning models, trained on
physically-simulated orbital/thermal data, can accurately predict component temperature
and classify overheat risk, offering a fast, data-driven complement to traditional
thermal analysis.

## Scope of the Project

- Generate a synthetic, physics-grounded dataset of orbital and thermal-design
  parameters and the corresponding steady-state component temperature.
- Train and compare regression models to predict temperature (°C).
- Train a classification model to flag components at risk of overheating
  (temperature ≥ 85°C).
- Evaluate model performance using standard regression and classification metrics.
- Provide a simple CLI and Python API for generating predictions on new inputs.
- Out of scope: real satellite telemetry integration, transient (time-dependent)
  thermal modeling, and multi-node thermal network simulation — the project uses a
  single-node, steady-state approximation.

## Target Users

- Students and early-career space systems/thermal engineers learning how ML can
  support thermal analysis workflows.
- Course evaluators assessing understanding of applied AI/ML concepts
  (regression, classification, evaluation, feature importance).

## High-Level Features

1. Physics-based synthetic data generation (Module 1a).
2. Data validation, cleaning, and preprocessing (Module 1b).
3. Temperature prediction via regression, comparing Linear Regression and
   Random Forest (Module 2a).
4. Overheat anomaly detection via classification (Module 2b).
5. Model evaluation and visualization, including feature importance analysis
   (Module 3).
6. A command-line interface tying the full pipeline together, plus a reusable
   `ThermalPredictor` class for programmatic predictions.
