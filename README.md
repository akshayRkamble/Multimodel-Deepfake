# Multidisciplinary Deepfake Detection

This project detects manipulated media across multiple modalities: images, videos, audio, tabular features, and text similarity. The frontend is now a React application in `ui/`.

## Project Structure

```text
Multimodel/
  data/                 Dataset inputs and generated processed data
  models/               Saved model artifacts
  notebooks/            Exploration, training, and evaluation notebooks
  reports/              Generated model reports
  scripts/              Project automation scripts
  src/                  Python model, training, evaluation, and utility code
  tests/                Unit tests
  ui/                   React + Vite frontend
  train_models.py       Training entrypoint
  evaluate_models.py    Evaluation entrypoint
  run_project.py        Local ML workflow runner
```

## Prerequisites

- Python 3.9 or newer for the ML code.
- Node.js 18 or newer for the React frontend.
- Git for version control.

## Install

Install Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Install React dependencies:

```bash
cd ui
npm install
```

## Run The React Frontend

```bash
cd ui
npm run dev
```

Vite will print the local URL, usually `http://localhost:5173/`.

## Build The Frontend

```bash
cd ui
npm run build
```

The production build is written to `ui/dist/`.

## ML Workflow

Train models:

```bash
python train_models.py
```

Evaluate models:

```bash
python evaluate_models.py
```

Run the local project workflow:

```bash
python run_project.py
```

## Models

- CNN for image classification.
- Vision Transformer for image classification.
- Transformer for sequence-oriented inputs.
- SVM for baseline tabular classification.
- Bayesian model for probabilistic baseline classification.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License. See [LICENSE](LICENSE).
