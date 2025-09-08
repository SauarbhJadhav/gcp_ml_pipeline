End-to-End Time Series Forecasting on Google Cloud Platform: Iowa Liquor Sales

1. Overview
Welcome! This document provides a step-by-step guide for building and orchestrating an end-to-end machine learning pipeline on the Google Cloud Platform (GCP). This project is designed as a practical demonstration for MSc-IT students to understand the key components of a modern MLOps workflow.

We will build a time series forecasting model using a Holt-Winters algorithm to predict daily liquor sales. The pipeline will perform the following actions:

Data Ingestion: Fetch prepared data from a Google BigQuery table.
Model Training: Train a forecasting model using a Python script.
Containerization: Package the Python application into a Docker container.
Registry: Store the container image in Google Artifact Registry.
Orchestration: Automate the entire workflow using a DAG (Directed Acyclic Graph) in Google Cloud Composer (managed Apache Airflow).
Architecture
The high-level architecture of our pipeline can be visualized as follows:

BigQuery (Data Source) -> Cloud Composer (Orchestrator) -> Triggers a Kubernetes Pod -> Docker Container (from Artifact Registry) -> Runs Python Script (Fetches data, trains model, makes forecast) -> BigQuery (Stores Forecast)

2. Prerequisites
Before you begin, please ensure you have the following set up:

Google Cloud Platform (GCP) Account: A GCP account with billing enabled. You can use the https://cloud.google.com/free for this project.
A GCP Project: Create a new project in the GCP Console.
Google Cloud SDK (gcloud): https://cloud.google.com/sdk/docs/install.
Docker: [Install Docker Desktop](https://www.docker.com/products/docker-desktop/) on your local machine.
Python 3.8+: A local Python environment.
Basic Knowledge: A foundational understanding of SQL, Python, Docker, and command-line interfaces.

3. Step-by-Step Implementation
Step 1: Data Preparation in BigQuery
For this demonstration, we will use the public Iowa Liquor Sales dataset. We will create a processed table that aggregates the total number of bottles sold per day for a specific store and a few specific items.

Navigate to BigQuery: In the GCP Console, go to the BigQuery UI.

Create a Dataset: First, create a new dataset in your project called processed.

Run the Aggregation Query: Execute the following SQL query in the BigQuery editor. This query sums the bottles sold per day for our selected items and saves the result into a new table named daily_liquor_sales_summary inside your processed dataset.

Note: We are aggregating the data by day to create a single time series, which is required for our forecasting model.

-- This query creates a table with daily sales aggregated per item for a specific store.
-- This structure is ideal for multi-series forecasting.

CREATE OR REPLACE TABLE `[YOUR_PROJECT_ID].processed.daily_liquor_sales_by_item` AS (
  SELECT
    date AS sale_date,
    store_number,
    item_number,
    -- We need to aggregate in case there are multiple sales records for the same item on the same day
    SUM(bottles_sold) AS total_bottles_sold
  FROM
    `bigquery-public-data.iowa_liquor_sales.sales`
  WHERE
    -- Filter for a specific store
    store_number = '10268'
    -- Filter for a few specific, popular items to serve as our independent time series
    AND item_number IN ('64870', '36904', '64864')
  GROUP BY
    sale_date, store_number, item_number
  ORDER BY
    store_number, item_number, sale_date
);

Note: Replace [YOUR_PROJECT_ID] with your actual GCP Project ID.

Step 2: The Forecasting Python Script
Next, we will create a Python script that connects to BigQuery, fetches the aggregated sales data, trains a Holt-Winters model, and generates a forecast.

Create a file named forecast_model.py and add the following code.

# forecast_model.py

import os
import pandas as pd
from google.cloud import bigquery
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def run_forecast():
    """
    Main function to run the forecasting pipeline.
    - Fetches aggregated sales data from BigQuery.
    - Trains a Holt-Winters model.
    - Generates a 30-day forecast.
    - Saves the forecast back to a new BigQuery table.
    """
    # --- 1. Configuration ---
    # GCP Project ID is automatically inferred from the environment
    # where the code is running (e.g., a Cloud Composer worker).
    project_id = os.environ.get("GCP_PROJECT")
    source_table = f"{project_id}.processed.daily_liquor_sales_summary"
    destination_table = f"{project_id}.processed.daily_liquor_sales_forecasts"
    forecast_days = 30

    print(f"Starting forecast process for project: {project_id}")
    print(f"Source table: {source_table}")

    # --- 2. Fetch Data from BigQuery ---
    bq_client = bigquery.Client()
    sql_query = f"SELECT sale_date, total_bottles_sold FROM `{source_table}` ORDER BY sale_date"
    
    print("Fetching data from BigQuery...")
    df = bq_client.query(sql_query).to_dataframe()
    
    # Ensure data types are correct for time series modeling
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df.set_index('sale_date', inplace=True)
    
    # The data has daily frequency. We specify this for the model.
    df = df.asfreq('D')
    # Fill any missing days with 0 sales
    df['total_bottles_sold'].fillna(0, inplace=True)
    
    print(f"Data fetched successfully. Shape: {df.shape}")

    # --- 3. Train Holt-Winters Model ---
    # We use an additive model for trend and seasonality, as retail sales
    # often have weekly cycles (seasonality period of 7 days).
    print("Training Holt-Winters Exponential Smoothing model...")
    model = ExponentialSmoothing(
        df['total_bottles_sold'],
        trend='add',
        seasonal='add',
        seasonal_periods=7
    ).fit()
    print("Model training complete.")

    # --- 4. Generate Forecast ---
    print(f"Generating forecast for the next {forecast_days} days...")
    forecast = model.forecast(steps=forecast_days)
    
    # Format the forecast into a DataFrame for storage
    forecast_df = pd.DataFrame({
        'forecast_date': forecast.index,
        'predicted_bottles_sold': forecast.values
    })
    # Round the predictions to the nearest integer
    forecast_df['predicted_bottles_sold'] = forecast_df['predicted_bottles_sold'].round().astype(int)
    
    print("Forecast generated successfully.")
    print(forecast_df.head())

    # --- 5. Save Forecast to BigQuery ---
    print(f"Saving forecast to BigQuery table: {destination_table}")
    job_config = bigquery.LoadJobConfig(
        # Overwrite the table with new forecasts each time the pipeline runs
        write_disposition="WRITE_TRUNCATE",
        # Define the schema for the destination table
        schema=[
            bigquery.SchemaField("forecast_date", "DATE"),
            bigquery.SchemaField("predicted_bottles_sold", "INTEGER"),
        ],
    )

    job = bq_client.load_table_from_dataframe(
        forecast_df, destination_table, job_config=job_config
    )
    job.result()  # Wait for the job to complete

    print(f"Forecast data successfully loaded to {destination_table}.")

if __name__ == "__main__":
    run_forecast()
Step 3: Containerize the Application with Docker
We need to package our Python script and its dependencies into a Docker image.

Create requirements.txt: This file lists the Python libraries our script needs.

# requirements.txt
pandas
google-cloud-bigquery
statsmodels
db-dtypes
Create Dockerfile: This file contains the instructions to build our Docker image.

# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script into the container
COPY forecast_model.py .

# Define the command to run the application
CMD ["python", "forecast_model.py"]
Step 4: Push the Docker Image to Artifact Registry
Artifact Registry is GCP's recommended service for storing and managing container images.

Enable the Artifact Registry API:

gcloud services enable artifactregistry.googleapis.com
Create a Docker Repository: Choose a region (e.g., us-central1) and a name for your repository.

gcloud artifacts repositories create [REPO_NAME] \
    --repository-format=docker \
    --location=[REGION] \
    --description="Docker repository for liquor sales forecasting model"
Replace [REPO_NAME] and [REGION].

Configure Docker Authentication: This command configures your local Docker client to authenticate with Artifact Registry.

gcloud auth configure-docker [REGION]-docker.pkg.dev
Build and Tag the Docker Image: From your project directory (containing Dockerfile, forecast_model.py, and requirements.txt), run the build command.

# Define variables for convenience
export PROJECT_ID=[YOUR_PROJECT_ID]
export REGION=[REGION]
export REPO_NAME=[REPO_NAME]
export IMAGE_NAME=liquor-sales-forecaster
export IMAGE_TAG=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest

# Build the image
docker build -t ${IMAGE_TAG} .
Replace the placeholders with your values.

Push the Image to Artifact Registry:

docker push ${IMAGE_TAG}
You can now verify that your image is in the Artifact Registry UI in the GCP Console.

Step 5: Orchestrate with Cloud Composer
Cloud Composer is a fully managed Apache Airflow service. We will create a DAG to run our containerized job on a schedule.

Create a Cloud Composer Environment:

Navigate to Composer in the GCP Console.
Create a new environment (Composer 2 is recommended). This can take 20-30 minutes.
During creation, ensure the service account used by the environment has the necessary permissions:
BigQuery Data Editor (to read and write tables)
Kubernetes Engine Developer or Workload Identity User (to run pods)
Artifact Registry Reader (to pull the Docker image)
Create the DAG File: Create a file named forecasting_dag.py. This Python script defines the Airflow workflow.

# forecasting_dag.py

from __future__ import annotations

import datetime
from airflow.models.dag import DAG
from airflow.providers.google.cloud.operators.kubernetes_engine import GKEStartPodOperator

# --- 1. DAG Configuration ---
with DAG(
    dag_id="daily_liquor_sales_forecasting_pipeline",
    start_date=datetime.datetime(2023, 1, 1),
    # Run the DAG daily at midnight UTC.
    schedule_interval="0 0 * * *",
    catchup=False,
    tags=["forecasting", "gcp", "sales"],
    description="A DAG to run a daily time series forecast for liquor sales.",
) as dag:
    # --- 2. Task Definition ---
    # This task will spin up a Kubernetes Pod in the Composer environment's GKE cluster
    # and run our Docker container inside it.
    run_forecast_pod = GKEStartPodOperator(
        task_id="run_liquor_forecast_model_pod",
        # The name of the pod to create.
        name="liquor-forecast-pod",
        # The GCP project ID.
        project_id="[YOUR_PROJECT_ID]",
        # The location of the GKE cluster (same as your Composer environment).
        location="[COMPOSER_REGION]",
        # The name of the GKE cluster (find this in your Composer env details).
        cluster_name="[COMPOSER_GKE_CLUSTER_NAME]",
        # The namespace to run the pod in. 'default' is usually fine.
        namespace="default",
        # The full path to the Docker image in Artifact Registry.
        image="[REGION]-docker.pkg.dev/[YOUR_PROJECT_ID]/[REPO_NAME]/liquor-sales-forecaster:latest",
        # Environment variables to pass to the container.
        # The Python script uses GCP_PROJECT to construct table names.
        env_vars={
            "GCP_PROJECT": "[YOUR_PROJECT_ID]"
        },
        # Ensure the pod is deleted after the task completes.
        do_xcom_push=False,
    )

Important: Replace the following placeholders in the DAG file:

[YOUR_PROJECT_ID]
[COMPOSER_REGION] (e.g., us-central1)
[COMPOSER_GKE_CLUSTER_NAME] (Find this in the "GKE cluster" link on your Composer environment's details page).
The full image path from Step 4.
Upload the DAG to Composer:

In the GCP Console, go to your Composer environment's details page.
Click on the "DAGs Folder" link. This will open a Google Cloud Storage (GCS) bucket.
Upload your forecasting_dag.py file to this bucket. Airflow will automatically detect and load it within a few minutes.

4. Running the Pipeline and Verifying Results
Trigger the DAG:

Open the Airflow UI from your Composer environment page.
Find the daily_liquor_sales_forecasting_pipeline DAG in the list.
Un-pause the DAG using the toggle on the left.
To run it immediately, click the "Play" button on the right.
Monitor the Run:

Click on the DAG name to see the Grid View.
You can click on the running task (run_liquor_forecast_model_pod) and view its logs to see the output from our Python script.
Check the Results:

Once the DAG run is successful, navigate back to the BigQuery UI.
In your processed dataset, you should now see a new table named daily_liquor_sales_forecasts.
Query this table to see the 30-day forecast generated by your model!
SELECT * FROM `[YOUR_PROJECT_ID].processed.daily_liquor_sales_forecasts` ORDER BY forecast_date;

5. Conclusion and Next Steps
Congratulations! You have successfully built and deployed an end-to-end ML pipeline on GCP. You learned how to:

Prepare data in BigQuery.
Develop and containerize a model with Python and Docker.
Store your container in Artifact Registry.
Orchestrate the entire workflow with Cloud Composer and Kubernetes.