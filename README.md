# End-to-End MLOps: Multi-Series Forecasting on GCP

This repository provides a comprehensive, end-to-end MLOps pipeline for performing time series forecasting on the Google Cloud Platform (GCP). It serves as a practical, step-by-step guide to modern data engineering and machine learning workflows, focusing on automating the training and deployment of multiple forecasting models—one for each unique item in a dataset—with predictions stored in BigQuery.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Step-by-Step Implementation](#step-by-step-implementation)
  - [1. GCP Setup & Data Preparation](#1-gcp-setup--data-preparation)
  - [2. Forecasting Application](#2-forecasting-application)
  - [3. CI/CD Pipeline (GitHub Actions)](#3-cicd-pipeline-github-actions)
  - [4. Orchestration with Cloud Composer](#4-orchestration-with-cloud-composer)
  - [5. Running the Pipeline](#5-running-the-pipeline)
  - [6. Troubleshooting](#6-troubleshooting)
- [License](#license)

---

## Project Overview

The goal of this project is to automate the training of multiple time series forecasting models (one per item) and store their predictions in BigQuery. The pipeline leverages GitHub Actions for CI/CD and Cloud Composer (Airflow) for orchestration.

**Use Case:**  
Forecasting daily sales for multiple liquor items using the public Iowa Liquor Sales dataset.

**Key Technologies:**
- **Data Storage & Warehousing:** Google BigQuery
- **Modeling:** Python (pandas, statsmodels/Holt-Winters)
- **Containerization:** Docker
- **Artifact Management:** Google Artifact Registry
- **CI/CD Automation:** GitHub Actions
- **Workflow Orchestration:** Google Cloud Composer (Apache Airflow)

---

## Architecture

The pipeline follows a modern MLOps architecture, separating CI/CD from orchestration:

- **Code Push (GitHub) → GitHub Actions (CI/CD) → Builds & Pushes Docker Image (Artifact Registry)**
- **Cloud Composer (Orchestration) → Triggers Daily DAG → Runs Kubernetes Pod → Pulls Image & Runs Container → Fetches Data (BigQuery) → Trains Models & Predicts → Saves Forecasts (BigQuery)**

---

## Prerequisites

Before starting, ensure you have:

- **Google Cloud Platform (GCP) Account:** [Sign up](https://cloud.google.com/free)
- **GCP Project:** Create and note the Project ID
- **Google Cloud SDK (gcloud):** [Install](https://cloud.google.com/sdk/docs/install)
- **Docker:** [Install Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Python 3.8+**
- **GitHub Repository**
- **GitHub Personal Access Token (PAT):** For command-line Git operations ([Guide](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token))

---

## Step-by-Step Implementation

### 1. GCP Setup & Data Preparation

- **Enable Required APIs:**
  ```sh
  gcloud services enable bigquery.googleapis.com
  gcloud services enable artifactregistry.googleapis.com
  gcloud services enable composer.googleapis.com
  gcloud services enable iamcredentials.googleapis.com
  ```
- **Create BigQuery Dataset:**  
  In BigQuery UI, create a dataset named `processed`.

- **Create Processed Table:**  
  Use the provided SQL to aggregate daily sales per item for a specific store.  
  _Replace `[YOUR_PROJECT_ID]` with your actual GCP Project ID._

### 2. Forecasting Application

- **requirements.txt:**  
  List dependencies:
  ```
  pandas
  google-cloud-bigquery
  statsmodels
  db-dtypes
  ```

- **forecast_model.py:**  
  Python script for multi-series forecasting using Holt-Winters, reading from and writing to BigQuery.

- **Dockerfile:**  
  Containerizes the application for deployment.

### 3. CI/CD Pipeline (GitHub Actions)

- **Create Artifact Registry Repository:**  
  ```sh
  gcloud artifacts repositories create [REPO_NAME] \
      --repository-format=docker \
      --location=[REGION] \
      --description="Docker repository for forecasting model"
  ```
- **Create GCP Service Account:**  
  - Grant Artifact Registry Writer role.
  - Download JSON key.

- **Configure GitHub Secrets:**  
  - `GCP_SA_KEY`: Service account JSON
  - `GCP_PROJECT_ID`: GCP Project ID
  - `GCP_ARTIFACT_REGISTRY_REGION`: Artifact Registry region
  - `GCP_ARTIFACT_REGISTRY_REPO`: Repository name

- **GitHub Actions Workflow:**  
  Automates Docker build and push to Artifact Registry.

### 4. Orchestration with Cloud Composer

- **Create Cloud Composer Environment:**  
  - Grant required roles to Composer's service account.

- **Create DAG File:**  
  - Use `GKEStartPodOperator` to run the forecasting container daily.

- **Upload DAG:**  
  - Place `forecasting_dag.py` in Composer's DAGs folder (GCS bucket).

### 5. Running the Pipeline

- **Commit and Push Code:**  
  ```sh
  git add .
  git commit -m "Initial pipeline setup"
  git push origin main
  ```
- **Monitor CI/CD:**  
  - Check GitHub Actions for successful Docker image build and push.

- **Trigger and Monitor Orchestration:**  
  - Use Airflow UI to trigger and monitor DAG runs.

- **Verify Results:**  
  - Check BigQuery for the `daily_liquor_sales_forecasts` table.

### 6. Troubleshooting

- **Git 403 Forbidden Error:**  
  - Use a Personal Access Token (PAT) for GitHub authentication.
  - Remove old credentials from your OS credential manager.
  - [GitHub PAT Guide](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token)

---


---

```yaml
  # GitHub Actions workflow snippet
  docker build . --tag "${{ env.GCP_REGION }}-docker.pkg.dev/${{ env.GCP_PROJECT_ID }}/${{ env.GCP_REPO_NAME }}/${{ env.IMAGE_NAME }}:latest"
  docker build . --tag "${{ env.GCP_REGION }}-docker.pkg.dev/${{ env.GCP_PROJECT_ID }}/${{ env.GCP_REPO_NAME }}/${{ env.IMAGE_NAME }}:${{ github.sha }}"

- name: Push Docker image to Artifact Registry
  run: |
    docker push "${{ env.GCP_REGION }}-docker.pkg.dev/${{ env.GCP_PROJECT_ID }}/${{ env.GCP_REPO_NAME }}/${{ env.IMAGE_NAME }}:latest"
    docker push "${{ env.GCP_REGION }}-docker.pkg.dev/${{ env.GCP_PROJECT_ID }}/${{ env.GCP_REPO_NAME }}/${{ env.IMAGE_NAME }}:${{ github.sha }}"
```

---

## Step 4: Orchestration with Cloud Composer

Finally, we'll set up the daily job to run our container.

**Create a Cloud Composer Environment:**

In the GCP Console, navigate to Composer and create a new environment (Composer 2).
During creation, edit the Service Account permissions. Grant it the following roles:

- BigQuery Data Editor
- Kubernetes Engine Developer
- Artifact Registry Reader
- Service Account User (on itself)

**Create the DAG File:** Create a file named `forecasting_dag.py`.

```python
# forecasting_dag.py
from __future__ import annotations
import datetime
from airflow.models.dag import DAG
from airflow.providers.google.cloud.operators.kubernetes_engine import GKEStartPodOperator

# --- Replace with your specific details ---
GCP_PROJECT_ID = "[YOUR_PROJECT_ID]"
COMPOSER_REGION = "[COMPOSER_REGION]"
COMPOSER_GKE_CLUSTER_NAME = "[COMPOSER_GKE_CLUSTER_NAME]"
ARTIFACT_REGISTRY_REGION = "[ARTIFACT_REGISTRY_REGION]"
ARTIFACT_REGISTRY_REPO = "[ARTIFACT_REGISTRY_REPO]"
IMAGE_NAME = "liquor-sales-forecaster"
# --- End of user-specific details ---

IMAGE_PATH = f"{ARTIFACT_REGISTRY_REGION}-docker.pkg.dev/{GCP_PROJECT_ID}/{ARTIFACT_REGISTRY_REPO}/{IMAGE_NAME}:latest"

with DAG(
    dag_id="multi_item_liquor_sales_forecasting_pipeline",
    start_date=datetime.datetime(2023, 1, 1),
    schedule_interval="0 0 * * *", # Daily at midnight UTC
    catchup=False,
    tags=["forecasting", "gcp", "sales", "multi-series"],
) as dag:
    run_forecast_pod = GKEStartPodOperator(
        task_id="run_multi_series_forecast_pod",
        name="multi-series-liquor-forecast-pod",
        project_id=GCP_PROJECT_ID,
        location=COMPOSER_REGION,
        cluster_name=COMPOSER_GKE_CLUSTER_NAME,
        namespace="default",
        image=IMAGE_PATH,
        env_vars={"GCP_PROJECT": GCP_PROJECT_ID},
        do_xcom_push=False,
    )
```

> **Important:** Replace all the placeholder values at the top of the DAG file. You can find the `[COMPOSER_GKE_CLUSTER_NAME]` on your Composer environment's details page.

**Upload the DAG:** On your Composer environment's page, click the "DAGs Folder" link to open a GCS bucket. Upload your `forecasting_dag.py` file there.

---

## 5. Running the Pipeline

**Commit and Push:** Commit all your new files (`forecast_model.py`, `Dockerfile`, `.github/workflows/build-and-push.yml`, etc.) and push them to the main branch of your GitHub repository.

```sh
git add .
git commit -m "Initial pipeline setup"
git push origin main
```

**Monitor CI/CD:** Go to the Actions tab in your GitHub repository. You will see the "Build and Push" workflow running. It should complete successfully, pushing your image to Artifact Registry.

**Trigger and Monitor Orchestration:**

1. Open the Airflow UI from your Cloud Composer environment.
2. Find the `multi_item_liquor_sales_forecasting_pipeline` DAG.
3. Un-pause it and trigger it manually using the play button.
4. Monitor the run. You can view the logs from the pod to see the Python script's output.

**Verify Results:** Once the DAG run is successful, go to BigQuery. A new table named `daily_liquor_sales_forecasts` should exist in your processed dataset. Query it to see your predictions!

```sql
SELECT *
FROM `[YOUR_PROJECT_ID].processed.daily_liquor_sales_forecasts`
ORDER BY item_number, forecast_date;
```

---

## 6. Troubleshooting

**Git 403 Forbidden Error when Pushing:** This error means your local Git client is using outdated or incorrect credentials. GitHub requires a Personal Access Token (PAT) for command-line operations, not your password.

**Generate a PAT:** In GitHub, go to Settings > Developer settings > Personal access tokens > Tokens (classic). Generate a new token with the repo scope. Copy the token.

**Clear Old Credentials:**

- **Windows:** Go to Control Panel > Credential Manager > Windows Credentials and remove the entry for `git:https://github.com`.
- **macOS:** Open the "Keychain Access" app, search for github.com, and delete the entry.

**Try Pushing Again:** The next time you run `git push`, you will be prompted for your username and password. For the password, paste your new Personal Access Token.
