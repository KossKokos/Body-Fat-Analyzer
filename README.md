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

This project was designed to reflect the full path from data science work to an actual application:

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
- project structure prepared for containerisation and cleaner deployment

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
React Frontend (Vite)
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

Local infrastructure:
- PostgreSQL runs in Docker
- models are stored locally inside the project