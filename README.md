# Docker Setup Guide

## 1. Using Dockerfiles

### Model Training and Threshold Determination

Build the training image:

```bash
docker build -t train-image:latest -f src/Dockerfile .
````

Run the container (mounting the artifacts directory):

```bash
docker run -v ./artifacts:/app/artifacts train-image:latest
```

This step trains the model and generates the required artifacts.

---

### Application (FastAPI + Uvicorn + Streamlit)

Build the application image:

```bash
docker build -t dev-app:latest -f dev/Dockerfile .
```

Run the application container:

```bash
docker run -p 5000:5000 -p 8501:8501 dev-app:latest
```

* FastAPI (Uvicorn) runs on port **5000**
* Streamlit runs on port **8501**

---

## 2. Using Docker Compose

Docker Compose allows you to manage multiple services together : Here (training, API, frontend) with a single configuration file.

```bash
docker compose up --build
```

