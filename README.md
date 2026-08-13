# [Body Fat Percentage Predictor](https://body-fat-analyzer.onrender.com/)

A full-stack machine learning portfolio project that takes a trained prediction pipeline and turns it into a working web application.

The app estimates body fat percentage from user-entered health, fitness, and nutrition data. It combines a TensorFlow-based prediction pipeline with a FastAPI backend, a PostgreSQL database, and a React frontend.

> **Important note**  
> This is a portfolio project for demonstration purposes only.  
> It is not a medical product and does not provide medical advice, diagnosis, or treatment.  
> Any result returned by the app is an estimate and should not be treated as a clinical measurement.

---

## Quick links

- [Live project](#live-project)
- [Why I built this project](#why-i-built-this-project)
- [What this project proves](#what-this-project-proves)
- [Machine learning approach](#machine-learning-approach)
- [Evaluation results](#evaluation-results)
- [Tech stack](#tech-stack)
- [Local setup](#local-setup)
- [Docker](#docker)
- [Deployment](#deployment)
- [Security-minded decisions](#security-minded-decisions)
- [Portfolio context](#portfolio-context)

---

## Live project

- **Frontend:** https://body-fat-analyzer.onrender.com/
- **Backend:** hosted as a Docker container on Koyeb
- **Database:** Neon PostgreSQL

The hosted version uses a split deployment:

```text
Render Static Site
  └── React frontend
        │
        ▼
Koyeb Web Service
  └── FastAPI backend running a Docker image from GitHub Container Registry
        │
        ▼
Neon PostgreSQL
```

---

## Why I built this project

I wanted to build something that shows more than just model training in a notebook.

This project was built to reflect the full path from data science work to an actual working application:

- working with a real dataset
- analysing, cleaning, and preparing the data
- building a reusable machine learning prediction pipeline
- combining classification and regression in one flow
- integrating trained models into a backend service
- storing prediction and feedback data in a relational database
- exposing the functionality through an API
- building a frontend that lets a user interact with the system
- thinking about validation, security, persistence, Docker, and deployment

For me, the value of this project is that it connects machine learning work with backend engineering and product thinking, instead of leaving the model isolated in a notebook.

---

## What this project proves

### Data science / machine learning

- practical work with structured lifestyle and health-related data
- dataset reading, analysis, cleaning, and preparation
- feature handling and reusable pipeline design
- combining **classification + regression** in one prediction workflow
- moving from training logic to backend-ready inference logic
- using trained TensorFlow models inside a running API service
- thinking beyond experiments toward a deployable ML application

### Backend and data engineering

- building a prediction API with **FastAPI**
- loading ML models once at application startup and reusing them for requests
- integrating trained models into an application service layer
- designing relational tables for prediction history and user feedback
- handling schema changes with **Alembic migrations**
- working with **PostgreSQL**
- using SQLAlchemy for database access
- implementing validation, error handling, and security-minded API behavior

### Frontend and full-stack integration

- separate frontend and backend services
- React form handling and validation
- prediction result rendering
- feedback prompt and modal flow
- loading, success, error, and reset states
- connecting a React UI to a machine learning backend through API requests
- clear portfolio/legal pages explaining the project is for demonstration only

### Deployment and DevOps

- local Docker Compose setup for frontend, backend, and PostgreSQL
- backend Docker image built and pushed to **GitHub Container Registry**
- backend deployed on **Koyeb** from a pre-built Docker image
- frontend deployed as a **Render Static Site**
- database hosted on **Neon PostgreSQL**
- environment-based configuration for local and hosted environments
- platform-specific database SSL handling
- practical debugging of hosted container issues

---

## Machine learning approach

The prediction pipeline is intentionally structured rather than flat.

At a high level:

1. user input is validated and prepared
2. features are ordered and transformed consistently for inference
3. a **classifier** predicts a body-fat class: `low`, `mid`, or `high`
4. the selected class routes the input into a class-specific regression flow
5. the final prediction is returned through the API
6. prediction history can be stored in PostgreSQL
7. optional user feedback can also be stored for future review and possible model improvement

This structure reflects an important part of the project: thinking about ML systems as reusable pipelines that can run inside a backend application, not just inside a notebook.

### Final model pipeline

The backend uses one final artifact bundle trained in
`loading_script.ipynb`. Its ordinal classifier respects the ordered
`low` → `mid` → `high` classes, while boundary weighting gives difficult
examples near the class thresholds more influence during training. The
selected class routes each prediction to its final class-specific base model
and residual regressor; the mid-fat regressor receives additional boundary
weighting.

Inference reindexes encoded and engineered inputs into the exact 57-feature
training contract. The request and response API remains unchanged.

The final artifacts use a newer Keras serialization format. If the installed
runtime cannot deserialize that format directly, the compatibility loader
reconstructs the same final architectures and loads their saved weights.


### Evaluation results

Final evaluation was produced by `loading_script.ipynb`.
The final test protocol uses one shared holdout cohort for every metric:

- source dataset: `back_end/app/ml/Final_data.csv`
- complete dataset size: 20,000 records
- development cohort: 18,000 records
- shared final test cohort: 2,000 records
- class thresholds learned from development data only: `low < 23.5562`, `23.5562 <= mid <= 28.1110`, `high > 28.1110`
- final feature contract after encoding and feature engineering: 57 features

Final test class distribution:

| Class | Development rows | Final test rows |
|---|---:|---:|
| Low | 5,940 | 648 |
| Mid | 5,940 | 661 |
| High | 6,120 | 691 |

Final holdout performance:

| Model | Metric | Score | Unit | Test rows |
|---|---|---:|---|---:|
| Ordinal classifier | Accuracy | 0.7890 | proportion correct | 2,000 |
| Low-fat hybrid regressor | MAE | 1.5689 | body-fat percentage points | 648 |
| Mid-fat hybrid regressor | MAE | 0.9421 | body-fat percentage points | 661 |
| High-fat hybrid regressor | MAE | 1.2676 | body-fat percentage points | 691 |
| Complete classifier-routed pipeline | MAE | 1.8287 | body-fat percentage points | 2,000 |

Classifier detail on the shared final test cohort:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Low | 0.8622 | 0.7824 | 0.8204 | 648 |
| Mid | 0.6554 | 0.7655 | 0.7062 | 661 |
| High | 0.8828 | 0.8177 | 0.8490 | 691 |

Classifier confusion matrix:

| Actual / Predicted | Low | Mid | High |
|---|---:|---:|---:|
| Low | 507 | 140 | 1 |
| Mid | 81 | 506 | 74 |
| High | 0 | 126 | 565 |

The classifier is an ordinal neural network, not a flat softmax classifier.
It predicts ordered cumulative class-boundary probabilities for
`low -> mid -> high`. Training uses balanced class weights plus additional
boundary weighting around the learned low/mid and mid/high thresholds.

Classifier training split:

| Split | Rows |
|---|---:|
| Train | 16,200 |
| Validation | 1,800 |
| Shared final test | 2,000 |

The regressor branch is a hybrid setup for each class: a Lasso base model
predicts the class-specific body-fat percentage, then a neural residual
regressor corrects the base prediction. The low and high residual models use
quantile losses, while the mid-fat residual model uses Huber loss with extra
boundary weighting.

Regressor training split:

| Class | Train rows | Validation rows | Shared final test rows |
|---|---:|---:|---:|
| Low | 5,346 | 594 | 648 |
| Mid | 5,346 | 594 | 661 |
| High | 5,508 | 612 | 691 |

---

## Architecture overview

```text
User
  │
  ▼
React Frontend
  │
  │  POST /api/predict/
  │  POST /api/feedback/
  ▼
FastAPI Backend
  │
  ├── Request validation
  ├── Prediction service
  │     ├── feature preparation
  │     ├── classifier
  │     ├── class-specific base model
  │     └── class-specific residual regressor
  │
  ├── SQLAlchemy / PostgreSQL
  │     ├── prediction_history
  │     └── prediction_feedback
  │
  └── Alembic migrations
```

---

## Tech stack

### Data science / model development

- Python
- pandas
- NumPy
- matplotlib
- seaborn
- TensorFlow
- statsmodels
- scikit-learn

### Backend

- FastAPI
- SQLAlchemy
- PostgreSQL
- Alembic
- pg8000
- Pydantic
- TensorFlow Docker image
- Docker

### Frontend

- React
- JavaScript
- Vite
- React Hook Form
- Yup
- Axios
- Tailwind CSS
- Nginx for the containerised frontend build

### Deployment

- Render Static Site for the frontend
- Koyeb Web Service for the backend container
- GitHub Container Registry for the backend Docker image
- Neon PostgreSQL for the hosted database
- Docker Compose for local containerised development

---

## Dataset

This project uses a public dataset from Kaggle:

**Lifestyle Data**  
https://www.kaggle.com/datasets/jockeroika/life-style-data

---

## Model training

Model training and experimentation are documented in:

```text
loading_script.ipynb
```

The notebook covers the data science and model development side of the project. The application code shows how that model work is turned into a reusable full-stack system.

---

## Completed features

- prediction form with validation
- backend prediction endpoint
- reusable model loading and inference pipeline
- prediction history persistence
- feedback submission flow
- feedback storage in PostgreSQL
- delayed feedback prompt after prediction
- star rating inside the feedback form
- result display and result download
- new prediction reset flow
- route-based frontend navigation
- About, Privacy Policy, and Terms pages
- API key protected backend routes
- minimal health check behavior
- environment-based configuration
- local Docker Compose setup
- hosted split deployment with Render, Koyeb, and Neon

---

## Repository structure

```text
Fitness_Proj
├── .gitignore
├── README.md
├── docker-compose.yml
├── .env.docker.example
├── loading_script.ipynb
│
├── back_end
│   ├── .env.example
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── requirements.txt
│   └── app
│       ├── alembic
│       ├── api
│       ├── config
│       ├── core
│       ├── database
│       ├── docker
│       ├── ml
│       ├── models
│       ├── services
│       ├── tests
│       ├── utils
│       └── main.py
│
└── front_end
    ├── .env.example
    ├── Dockerfile
    ├── .dockerignore
    ├── nginx.conf
    ├── package.json
    ├── vite.config.js
    ├── public
    └── src
        ├── api
        ├── components
        ├── pages
        └── utils
```

---

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/KossKokos/Body-Fat-Analyzer
cd Fitness_Proj
```

### 2. Backend setup without Docker

```bash
cd back_end
pip install -r requirements.txt
```

Create a backend environment file:

```text
back_end/.env
```

You can start from:

```text
back_end/.env.example
```

Run the backend from `back_end/app`:

```bash
py main.py
```

Or from the repository root:

```bash
py back_end/app/main.py
```

### 3. Frontend setup without Docker

```bash
cd front_end
npm install
npm run dev
```

Create a frontend environment file:

```text
front_end/.env
```

You can start from:

```text
front_end/.env.example
```

---

## Docker

The project can be run locally as a fully containerised stack using Docker Compose.

The local Docker setup includes:

- **PostgreSQL** as a dedicated database container
- **FastAPI backend** as the API and model inference service
- **React frontend** built with Vite and served through **Nginx**
- **Alembic migrations** applied automatically during backend startup

### Container structure

```text
docker-compose.yml
  ├── db
  │   └── PostgreSQL 16
  │
  ├── backend
  │   └── FastAPI + TensorFlow + Alembic
  │
  └── frontend
      └── React build served by Nginx
```

### Docker environment configuration

Docker uses a dedicated environment file:

```text
.env.docker
```

A safe template version is included:

```text
.env.docker.example
```

Real secret values should stay only in local environment files and should not be committed.

### Local Docker database driver

The backend uses `pg8000` with SQLAlchemy.

For local Docker PostgreSQL, SSL should be disabled:

```env
DB_SSL_MODE=disable
SQLALCHEMY_DATABASE_URL=postgresql+pg8000://USER:PASSWORD@db:5432/DB_NAME
```

Do not add `?sslmode=require` to the local Docker database URL.

### Run with Docker

From the project root:

```bash
docker compose --env-file .env.docker up --build
```

Default local ports:

- **Frontend:** `http://localhost:8080`
- **Backend:** `http://localhost:8000`
- **PostgreSQL:** `localhost:5433`

### Backend startup inside Docker

Inside Docker, the backend is not started with the local Windows command:

```bash
py back_end/app/main.py
```

Instead, the backend container uses an entrypoint script that:

1. waits for PostgreSQL to become available
2. applies Alembic migrations
3. starts the FastAPI app with Uvicorn

This makes startup more reliable in containerised environments.

---

## Deployment

The final hosted project uses a split deployment.

```text
Render Static Site
  └── React frontend
        │
        ▼
Koyeb Web Service
  └── FastAPI backend container from GitHub Container Registry
        │
        ▼
Neon PostgreSQL
```

### Frontend deployment

The frontend is deployed on **Render** as a static site.

The frontend calls the backend through:

```env
VITE_API_BASE_URL=https://YOUR-KOYEB-BACKEND.koyeb.app
```

Vite environment variables are build-time values, so the frontend must be redeployed after changing them.

### Backend deployment

The backend is packaged as a Docker image, pushed to **GitHub Container Registry**, and deployed on **Koyeb**.

This approach gives more control over the exact backend image being deployed and avoids differences between local Docker builds and platform-side builds.

Example image format:

```text
ghcr.io/kosskokos/body-fat-backend:pg8000-v3
```

Versioned image tags are preferred over relying only on `latest`, because they make it clear exactly which image Koyeb is running.

### Hosted backend environment

On Koyeb, the backend uses:

```env
DB_SSL_MODE=require
SQLALCHEMY_DATABASE_URL=postgresql+pg8000://USER:PASSWORD@DIRECT_NEON_HOST:5432/DB_NAME
```

Do not add `?sslmode=require` to the `pg8000` URL. SSL is handled in the backend through SQLAlchemy `connect_args` when `DB_SSL_MODE=require`.

### Database deployment

The hosted database is **Neon PostgreSQL**.

Important database notes:

- local Docker PostgreSQL uses `DB_SSL_MODE=disable`
- Neon PostgreSQL uses `DB_SSL_MODE=require`
- the hosted backend uses the direct Neon host
- Alembic migrations run during backend startup
- `pg8000` is used to avoid native PostgreSQL driver issues in the hosted container environment

---

## Testing

Backend test scripts are located in:

```text
back_end/app/tests/
```

Available scripts include:

- `health_test.py`
- `predict_test.py`
- `feedback_test.py`
- `test_model_pipeline_contracts.py`
- `test_model_artifact_smoke.py`

These scripts were used during development to verify:

- health endpoint behavior
- database health behavior
- prediction request flow
- feedback request flow
- invalid API key behavior

The model contract tests run without loading the heavy artifacts. To load the
final artifact bundle and make a real pipeline prediction:

```powershell
cd back_end/app
$env:RUN_MODEL_ARTIFACT_SMOKE="1"
..\.venv\Scripts\python.exe -m unittest tests.test_model_artifact_smoke -v
```

You can also test deployed endpoints directly using PowerShell.

Example health request:

```powershell
Invoke-WebRequest `
  -Uri "https://YOUR-BACKEND.koyeb.app/api/health/" `
  -Method GET `
  -Headers @{ "X-API-Key" = "YOUR_API_KEY" }
```

Example database health request:

```powershell
Invoke-WebRequest `
  -Uri "https://YOUR-BACKEND.koyeb.app/api/health/db" `
  -Method GET `
  -Headers @{ "X-API-Key" = "YOUR_API_KEY" }
```

---

## Security-minded decisions

This project includes several practical security and application-quality decisions:

- API key checks on backend routes
- strict request validation with Pydantic
- controlled backend error responses
- environment-based configuration instead of hard-coded secrets
- real secrets excluded from Git
- minimal health endpoint responses
- separate frontend and backend services
- awareness that frontend-exposed API keys are not real security on their own
- database-backed persistence instead of temporary in-memory-only flows
- SSL required for the hosted Neon database connection
- clear user-facing disclaimer that the app is a portfolio project, not a medical product

This is still a portfolio project, but the structure was built to reflect realistic engineering decisions instead of only making the happy path work.

---

## Environment variables

Real environment files should not be committed.

Useful templates:

```text
back_end/.env.example
front_end/.env.example
.env.docker.example
```

Important hosted backend values:

```env
ALLOWED_ORIGINS=["https://body-fat-analyzer.onrender.com"]
ALLOWED_HOSTS=["YOUR-KOYEB-BACKEND.koyeb.app"]

DB_SSL_MODE=require
SQLALCHEMY_DATABASE_URL=postgresql+pg8000://USER:PASSWORD@DIRECT_NEON_HOST:5432/DB_NAME
```

Important local Docker values:

```env
DB_SSL_MODE=disable
SQLALCHEMY_DATABASE_URL=postgresql+pg8000://USER:PASSWORD@db:5432/DB_NAME
```

---

## What I would improve next

- separate training-only assets from runtime backend assets more strictly
- add deeper model evaluation reporting to the repository
- add structured monitoring around prediction and feedback usage
- improve CI checks for backend and frontend builds
- add optional authentication if the project ever evolves beyond portfolio/demo use

---

## Portfolio context

This project is hosted as a portfolio piece so employers and reviewers can see the system working as a real application rather than only reading about the model.

It is meant to show:

- practical data science thinking
- reusable ML pipeline design
- backend integration of machine learning
- relational database awareness
- API and frontend integration
- full-stack engineering discipline
- Docker and containerisation awareness
- practical deployment problem-solving

---

## Links

- **Live demo:** https://body-fat-analyzer.onrender.com/
- **LinkedIn:** [ADD LINKEDIN LINK]
- **Email:** [ADD YOUR EMAIL]

---

## Author

**Kostiantyn Pereimybida**

---

## Disclaimer

This project is for portfolio and demonstration purposes only.

It is not a medical application and should not be used for diagnosis, treatment, or healthcare decision-making.
