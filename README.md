# Body Fat Percentage Predictor

A full-stack machine learning project built to take a prediction pipeline beyond model training and turn it into a working application.

This project estimates body fat percentage from user-entered health, fitness, and nutrition data. It combines a machine learning pipeline with a FastAPI backend, a PostgreSQL database, and a React frontend, so the result is not just a trained model, but a complete usable system.

> **Important note**  
> This is a portfolio project for demonstration purposes only.  
> It is not a medical product and does not provide medical advice, diagnosis, or treatment.  
> Any result returned by the app is an estimate and should not be treated as a clinical measurement.

---

## Why I built this project

I wanted to build something that shows more than just model training in a notebook.

This project was built to reflect the full path from data science work to an actual application:

- working with a real dataset
- analysing, cleaning, and preparing the data
- building a reusable ML prediction pipeline
- combining classification and regression in one flow
- integrating trained models into a backend service
- storing prediction and feedback data in a relational database
- exposing the functionality through an API
- building a frontend that lets a user interact with the system
- thinking about validation, security, persistence, and deployment structure

For me, the value of the project is that it connects machine learning work with backend engineering and product thinking, instead of leaving the model isolated in a notebook.

---

## What this project proves

### Data science / machine learning
- practical work with structured lifestyle and health-related data
- dataset reading, analysis, cleaning, and preparation
- feature handling and reusable pipeline design
- combining **classification + regression** in one prediction workflow
- moving from training logic to backend-ready inference logic
- thinking beyond experiments toward a deployable ML application

### Backend and data engineering
- building a prediction API with **FastAPI**
- integrating trained models into an application service layer
- designing relational tables for prediction history and user feedback
- handling schema evolution with **Alembic migrations**
- working with **PostgreSQL**
- implementing validation, error handling, and security-minded API behavior

### Frontend and full-stack integration
- separate frontend and backend services
- form validation and controlled UI flows
- result rendering and feedback submission
- handling loading, success, and error states cleanly
- connecting a React UI to a machine learning backend through an API

### Engineering and deployment awareness
- environment-based configuration
- API key handling for protected endpoints
- separation of concerns across frontend, backend, database, and ML services
- Docker-based local PostgreSQL setup
- full containerisation with Docker Compose
- project structure prepared for cleaner local setup and deployment

---

## Machine learning approach

The prediction pipeline is intentionally structured rather than flat.

At a high level:

1. user input is validated and prepared
2. features are ordered and transformed consistently for inference
3. a **classifier** predicts a body-fat class (`low`, `mid`, `high`)
4. the selected class routes the input into a class-specific regression flow
5. the final prediction is returned through the API
6. prediction history can be stored in PostgreSQL
7. optional user feedback can also be stored for future review and possible model improvement

This structure reflects an important part of the project: thinking about ML systems as pipelines that can be reused in a backend environment, not just trained once in a notebook.

---

## Architecture overview

```text
User
  │
  ▼
React Frontend (Vite / Nginx)
  │
  │  POST /api/predict/
  │  POST /api/feedback/
  ▼
FastAPI Backend
  │
  ├── Input validation / request handling
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
- numpy
- matplotlib
- seaborn
- TensorFlow
- statsmodels

### Backend
- FastAPI
- SQLAlchemy
- PostgreSQL
- Alembic
- psycopg2
- Pydantic
- Docker

### Frontend
- React
- JavaScript
- Vite
- React Hook Form
- Yup
- Axios
- Tailwind CSS
- Nginx (for the containerised frontend build)

---

## Dataset

This project uses a public dataset from Kaggle:

**Lifestyle Data**  
https://www.kaggle.com/datasets/jockeroika/life-style-data

---

## Model training

Model training and experimentation are documented in:

```text
model_training.ipynb
```

This notebook covers the model development side of the project, while the application code shows how that work is turned into a reusable full-stack system.

---

## Completed features

- prediction form with validation
- backend prediction endpoint
- reusable model loading and inference pipeline
- prediction result display
- prediction history persistence
- optional feedback flow after prediction
- feedback storage in the database
- user-facing About, Privacy Policy, and Terms pages
- clean result download/export
- new prediction reset flow
- route-based frontend navigation
- API key protected backend routes
- minimal health check behavior
- environment-based configuration
- full Docker Compose setup for frontend, backend, and database

---

## Repository structure

```text
Fitness_Proj
├── .gitignore
├── back_end
│   ├── .env.example
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── app
│   │   ├── alembic
│   │   ├── api
│   │   ├── config
│   │   ├── core
│   │   ├── database
│   │   ├── docker
│   │   ├── main.py
│   │   ├── ml
│   │   ├── models
│   │   ├── services
│   │   ├── tests
│   │   └── utils
│   └── requirements.txt
├── front_end
│   ├── .env.example
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── nginx.conf
│   ├── public
│   ├── src
│   │   ├── api
│   │   ├── components
│   │   ├── pages
│   │   └── utils
│   ├── package.json
│   └── vite.config.js
├── docker-compose.yml
├── .env.docker.example
└── model_training.ipynb
```

---

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/KossKokos/Body-Fat-Analyzer
cd Fitness_Proj
```

---

### 2. Backend setup

Move into the backend project:

```bash
cd back_end
```

Create and activate a virtual environment if needed, then install dependencies:

```bash
pip install -r requirements.txt
```

#### Backend environment file

Create a `.env` file in:

```text
back_end/.env
```

You can start from `.env.example`.

#### Run the backend

From `back_end/app`:

```bash
py main.py
```

Or from the repository root:

```bash
py back_end/app/main.py
```

---

### 3. Frontend setup

Move into the frontend project:

```bash
cd front_end
```

Install dependencies:

```bash
npm install
```

#### Frontend environment file

Create a `.env` file in:

```text
front_end/.env
```

You can start from `.env.example`.

#### Run the frontend

```bash
npm run dev
```

---

### 4. Database

PostgreSQL can be run through Docker in local development.

---

## Docker

The project can also be run as a fully containerised stack using Docker Compose.

This setup includes:

- **PostgreSQL** as a dedicated database container
- **FastAPI** as the backend API and model inference service
- **React frontend** built with Vite and served through **Nginx**
- **Alembic migrations** applied automatically during backend startup

### Why Docker was added

This project originally worked outside full containerisation. Docker was added to make the application easier to start, easier to share, and closer to a real deployment workflow.

Containerising the stack also helped with:

- keeping the services clearly separated
- avoiding hidden local dependencies
- making database setup more consistent
- running migrations in a predictable way
- showing deployment awareness as part of the project

### Container structure

The Docker setup uses three services:

- **db** → PostgreSQL database
- **backend** → FastAPI application with model loading and Alembic migrations
- **frontend** → production build of the React app served by Nginx

### Docker environment configuration

Docker uses a dedicated environment file:

```text
.env.docker
```

A safe template version is included in the repository:

```text
.env.docker.example
```

Real secret values should be stored only in the local `.env.docker` file and should not be committed.

### Run with Docker

From the project root:

```bash
docker compose --env-file .env.docker up --build
```

### Default ports

- **Frontend:** `http://localhost:8080`
- **Backend:** `http://localhost:8000`
- **PostgreSQL:** `localhost:5433`

### Notes about backend startup

Inside Docker, the backend is not started with the local Windows command:

```bash
py back_end/app/main.py
```

Instead, the container starts the backend using a Linux-friendly runtime command through an entrypoint script. That entrypoint:

1. waits for PostgreSQL to become available
2. applies Alembic migrations
3. starts the FastAPI app with Uvicorn

This makes startup more reliable and better suited for containerised environments.

### What Docker adds to the project

Adding Docker to the project shows that the application is not only built and working locally, but can also be packaged into a cleaner and more portable environment.

From a portfolio point of view, this helps show:

- deployment awareness
- container-based service separation
- reproducible local setup
- cleaner database handling
- more realistic full-stack project structure


## Testing

Backend test scripts are located in:

```text
back_end/app/tests/
```

Available scripts include:
- `health_test.py`
- `predict_test.py`
- `feedback_test.py`

These are lightweight endpoint checks used during development to verify:
- health endpoint behavior
- prediction request flow
- feedback request flow

---

## Security-minded decisions in the project

This project includes several practical security and application-quality decisions:

- API key checks on backend requests
- strict request validation with Pydantic
- controlled backend error responses
- environment-based configuration instead of hard-coded secrets
- minimal health endpoint responses
- separate frontend and backend services
- awareness that frontend-exposed API keys are not real security on their own
- database-backed persistence instead of temporary in-memory-only flows for prediction history and feedback

This is still a portfolio project, but I wanted the code structure to reflect realistic engineering decisions rather than only getting the happy path working.

---

## What I would improve next

- stronger separation of training artifacts vs application runtime assets
- deeper model evaluation reporting
- richer monitoring/logging around prediction usage
- optional authentication if the project evolves beyond portfolio use

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

---

## Links

- **Live demo:** [ADD LIVE DEMO LINK]
- **LinkedIn:** [ADD LINKEDIN LINK]
- **Email:** [ADD YOUR EMAIL]

---

## Author

**Kostiantyn Pereimybida**

---

## Disclaimer

This project is for portfolio and demonstration purposes only.

It is not a medical application and should not be used for diagnosis, treatment, or healthcare decision-making.