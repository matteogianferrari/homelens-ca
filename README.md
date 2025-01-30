# HomeLens CA

> **Predicting Median House Values in California (1990) Using Block-Level Information**

HomeLens CA is a system designed to predict median house values in California based on block-level information. By leveraging modern MLOps practices and a microservice architecture, the system provides a flexible and scalable infrastructure suitable for rapid experimentation and deployment.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Continuous Integration and Deployment](#continuous-integration-and-deployment)
- [Installation](#installation)
- [Docker](#docker)
- [Usage](#usage)
- [License](#license)

---

## Project Overview

HomeLens CA provides an infrastructure for:

1. **UI Microservice**: A Gradio-based interface allowing users to input housing features and obtain predictions.
2. **Model Microservice**: A REST API using a trained MLflow-registered model via FastAPI.
3. **MLOps Integration**: MLflow for experiment tracking, Docker for containerization, and GitHub Actions for CI/CD automation.
4. **Deployment Flexibility**: Deployed on Render with web hooks for automatic redeployment.

---

## Features

- **Microservices Architecture**\
  Separate UI and Model microservices ensure scalability and maintainability.
- **Docker-Based Deployment**\
  Fully containerized services with Docker and docker-compose.
- **CI/CD Integration**\
  Automated testing, linting, and deployment using GitHub Actions.
- **MLflow Model Tracking**\
  Experiments, model registry, and automated updates.
- **Cloud-Based Deployment**\
  Hosted on Render for free-tier deployment and scaling.

---

## System Architecture

The system follows a microservices approach:

1. **UI Microservice (Gradio Frontend)**
2. **Model Microservice (FastAPI + MLflow Model Registry)**
3. **CI/CD Pipeline (GitHub Actions, GitHub Packages)**
4. **Deployment (Render Web Services, Docker Containers)**

---

## Continuous Integration and Deployment

- **CI Pipeline**: Automated testing, linting, and Docker image building.
- **CD Pipeline**: Deploys new versions to Render when changes are pushed.
- **Model Pipeline**: MLflow updates automatically if a new model outperforms the existing one.

---

## Installation
If you want to modify the project:
1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-repository/homelens_ca.git
   cd homelens_ca
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Docker

- **Building and Running Services**
  ```bash
  docker-compose up --build
  ```
- **Linting Dockerfiles**
  ```bash
  hadolint Dockerfile
  ```

---

## Usage

You can access the hosted application on Render (being a free hosting platform, the microservices will spin down with inactivity, thus it requires some time to spin them back up): [HomeLens CA on Render](#homelens-ca-ui-w2zj.onrender.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

