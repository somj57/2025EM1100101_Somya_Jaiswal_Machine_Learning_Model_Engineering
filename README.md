# Regression: House Prices Prediction 
**Course Assignment:** Machine Learning Model Engineering  
**Student Name:** Somya Jaiswal  
**Student ID:** 2025EM1100101  

## Project Overview
This project implements a production-ready Machine Learning pipeline for a **Regression** task: predicting house prices. Following the requirements for a full ML lifecycle, this notebook covers:

*   **Data & Features:** Engineering non-trivial features and addressing training-serving skew.
*   **Model Training:** Implementing a repeatable pipeline with baseline vs. candidate evaluation.
*   **Serving:** Designing an inference pattern (FastAPI-ready).
*   **Monitoring:** Planning for drift detection and retraining triggers.

### Dataset
We are using the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) dataset from Kaggle.

### Architecture Diagram
```text
graph TD
    subgraph "Data Pipeline (Batch)"
        A[Daily CSV Drops] --> B[Ingestion Script]
        B --> C[(Raw Data Storage)]
    end

    subgraph "Training Pipeline"
        C --> D[Data Split]
        D --> E[Scikit-learn Pipeline\nFeature Eng + RF Model]
        E --> F[Offline Evaluation\nBaseline vs Candidate]
        F -- "If better" --> G[(Model Registry\nmodels/v1/)]
    end

    subgraph "Online Serving"
        H[End User Web App] -->|JSON Input| I[FastAPI Endpoint /predict]
        G -->|Load .joblib| I
        I -->|Predicted Price| H
    end

    subgraph "Monitoring & MLOps"
        I -->|Log Inputs/Outputs| J[Monitoring Logs]
        J --> K[Drift Check Script]
        K -- "Alert/Trigger" --> A
    end
```    

### Project Structure
```text
house_price_ml_system/
├── data/
│   ├── raw/                 # Downloaded CSVs
│   └── processed/           # Feature-engineered data
├── models/
│   └── v1/                  # Saved artifacts (.joblib, metrics.json)
├── src/
│   ├── ingest.py            # Data ingestion script
│   ├── features.py          # Feature engineering logic
│   ├── train.py             # Training & Evaluation pipeline
│   ├── serve.py             # FastAPI application
│   └── monitor.py           # Drift detection & retraining logic
├── tests/
│   └── test_api.py          # Latency & load testing script
├── requirements.txt         
├── Dockerfile               # (Optional but guarantees max marks)
└── Design_Document.pdf      # Your 4-6 page report
```

### Terminals Output

1. Setup environment 
```text
(base) somyajaiswal@somyas-MacBook-Air house_price_ml_system % pip install -r requirements.txt
```

2. Running Data Ingestion
```text
(base) somyajaiswal@somyas-MacBook-Air house_price_ml_system % python src/ingest.py
[2026-07-29 15:31:53.434199] Ingesting data from https://raw.githubusercontent.com/somj57/2025EM1100101_Somya_Jaiswal_Machine_Learning_Model_Engineering/refs/heads/master/house_pricing_data/train.csv...
[2026-07-29 15:31:56.727686] Ingested 1460 rows. Saved to data/raw/training_table.csv
```

3. Training the models
```text
(base) somyajaiswal@somyas-MacBook-Air house_price_ml_system % python src/train.py
Baseline RMSE: $87,619.03
Candidate RMSE: $32,600.14
Candidate model is better. Promoting to production!
```

4. Starting the Inference Service 
```text
(base) somyajaiswal@somyas-MacBook-Air house_price_ml_system % uvicorn src.serve:app --reload
INFO:     Will watch for changes in these directories: ['/Users/somyajaiswal/Desktop/BITS/TRI 3/Machine Learning Model Engineering/2025EM1100101_Somya_Jaiswal_Machine_Learning_Model_Engineering/house_price_ml_system']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [63886] using StatReload
INFO:     Started server process [63888]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

5. Testing api latency (using a new terminal window)
```text
(base) somyajaiswal@somyas-MacBook-Air house_price_ml_system % python tests/test_api.py
Total Requests: 100
Average Latency: 5.94 ms
P95 Latency: 14.11 ms
```

6. Simulating the Monitoring Drift Check
```text
(base) somyajaiswal@somyas-MacBook-Air house_price_ml_system % python -c "from src.monitor import check_data_drift; check_data_drift('data/raw/training_table.csv', 'data/raw/training_table.csv')"
Training Mean BsmtSF: 1057.43 | New Batch Mean: 1057.43
```





