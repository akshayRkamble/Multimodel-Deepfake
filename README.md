# Reality Shield - Deepfake Detection

Reality Shield detects manipulated media across images, videos, audio, CSV feature files, and text similarity workflows. The project has three local services:

- Flask detection API on `http://127.0.0.1:5000`
- Flask SQLite auth app on `http://127.0.0.1:5001`
- React + Vite frontend on `http://127.0.0.1:5173/index.html`

Start the auth app first in your browser. After login, it redirects to the React frontend.

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

## First-Time Setup

Open PowerShell in the project root:

```powershell
cd G:\Deepfake\Reality_Shield
```

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

Install React dependencies:

```powershell
cd ui
npm install
cd ..
```

## Start The Whole Project

Use three PowerShell terminals. Keep all three running while using the app.

### Terminal 1 - Start The Detection API

```powershell
cd G:\Deepfake\Reality_Shield
.\.venv\Scripts\Activate.ps1
python backend\app.py
```

API URL:

```text
http://127.0.0.1:5000
```

### Terminal 2 - Start The Login/Register App

```powershell
cd G:\Deepfake\Reality_Shield
.\.venv\Scripts\Activate.ps1
python backend\auth_app.py
```

Auth URL:

```text
http://127.0.0.1:5001
```

### Terminal 3 - Start The React Frontend

```powershell
cd G:\Deepfake\Reality_Shield\ui
npm run dev -- --host 127.0.0.1 --port 5173
```

Frontend URL:

```text
http://127.0.0.1:5173/index.html
```

## How To Use

1. Open `http://127.0.0.1:5001`.
2. Register a new account.
3. Log in with your username and password.
4. After login, you will be redirected to the main React project.
5. Upload image, video, audio, or CSV files in the detector section.

User accounts are saved in the SQLite database:

```text
backend/auth_users.db
```

This database is ignored by Git.

## Optional Environment Variables

Change the auth secret key:

```powershell
$env:SECRET_KEY="change-this-secret"
python backend\auth_app.py
```

Change the SQLite database path:

```powershell
$env:AUTH_DATABASE="G:\Deepfake\Reality_Shield\backend\auth_users.db"
python backend\auth_app.py
```

Change the page opened after login:

```powershell
$env:MAIN_PROJECT_URL="http://127.0.0.1:5173/index.html"
python backend\auth_app.py
```

## Build The Frontend

```powershell
cd ui
npm run build
```

The production build is written to `ui/dist/`.

## ML Workflow

Train models:

```powershell
python train_models.py
```

Evaluate models:

```powershell
python evaluate_models.py
```

Run the local ML workflow:

```powershell
python run_project.py
```

## Troubleshooting

- If login works but the main page does not open, make sure Terminal 3 is running.
- If uploads fail, make sure Terminal 1 is running on port `5000`.
- If `npm run dev` fails, run `npm install` inside `ui/` again.
- If Flask says a port is already in use, stop the old Python process or restart your terminal.
- If PowerShell blocks virtual environment activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Models

- CNN for image classification.
- Vision Transformer for image classification.
- Transformer for sequence-oriented inputs.
- SVM for baseline tabular classification.
- Bayesian model for probabilistic baseline classification.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License. See [LICENSE](LICENSE).
