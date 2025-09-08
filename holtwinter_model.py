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
    project_id = "formidable-feat-466408-r6"
    source_table = f"{project_id}.processed.liquor_sales"
    destination_table = f"{project_id}.processed.daily_liquor_sales_forecasts"
    forecast_days = 30
    seasonal_periods = 7
    
    print(f"Starting forecast process for project: {project_id}")
    print(f"Source table: {source_table}")
    
    bq_client = bigquery.Client()
    sql_query = f"""
        SELECT
            sale_date,
            item_number,
            SUM(bottles_sold) as total_bottles_sold
        FROM
            `{source_table}`
        GROUP BY
            sale_date, item_number
        ORDER BY
            item_number, sale_date
    """
    
    print("Fetching data from BigQuery...")
    df = bq_client.query(sql_query).to_dataframe()
    
    all_forecasts = []
    
    # Group by the unique identifiers for each time series
    grouped = df.groupby(['item_number'])
    
    print(f"Found {len(grouped)} unique time series to process.")
    
    for (item,), series_df in grouped:
        print(f"--- Processing series for Item: {item} ---")
        
        # Prepare the individual time series
        series_df = series_df.set_index('sale_date').sort_index()
        
        # Create a complete daily time series, filling missing days with 0 sales.
        ts = series_df['total_bottles_sold'].asfreq('D').fillna(0)
        
        # A simple check to ensure we have enough data to train a model
        if len(ts) < 2 * seasonal_periods:
            print(f"Warning: Skipping series for Item {item} due to insufficient data ({len(ts)} points).")
            continue
            
        # Train Holt-Winters Model
        print(f"Training model on {len(ts)} data points...")
        try:
            model = ExponentialSmoothing(
                ts,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods
            ).fit()

            # Generate Forecast
            forecast = model.forecast(steps=forecast_days)
            
            # Format the forecast into a DataFrame
            forecast_df = pd.DataFrame({
                'forecast_date': forecast.index,
                'item_number': item,
                'predicted_bottles_sold': forecast.values.round().astype(int)
            })
            
            all_forecasts.append(forecast_df)
            print("Forecast generated and collected.")
        
        except Exception as e:
            print(f"Error processing series for Item {item}: {e}")
            continue
        
    if not all_forecasts:
        print("No forecasts were generated. Exiting.")
        return
    
    # --- 4. Combine and Save All Forecasts to BigQuery ---
    final_forecast_df = pd.concat(all_forecasts, ignore_index=True)
    print(f"All forecasts combined. Total predicted rows: {len(final_forecast_df)}")
    
    print(f"Saving all forecasts to BigQuery table: {destination_table}")
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[
            bigquery.SchemaField("forecast_date", "DATE"),
            bigquery.SchemaField("item_number", "STRING"),
            bigquery.SchemaField("predicted_bottles_sold", "INTEGER"),
        ],
    )
    job = bq_client.load_table_from_dataframe(
        final_forecast_df, destination_table, job_config=job_config
    )
    
    
    job.result()  # Wait for the job to complete

    print(f"All forecast data successfully loaded to {destination_table}.")
    
if __name__ == "__main__":
    run_forecast()