import requests
import time
import numpy as np

def measure_latency():
    url = "http://localhost:8000/predict"
    payload = {
        "customerID": "TEST_01",
        "tenure": 12,
        "MonthlyCharges": 75.5,
        "SeniorCitizen": 0,
        "InternetService": "Yes",
        "PhoneService": "Yes",
        "Contract": "Month-to-month"
    }
    
    latencies = []
    print("Sending 100 sequential requests to /predict endpoint...")
    
    for _ in range(100):
        start = time.time()
        resp = requests.post(url, json=payload)
        end = time.time()
        
        if resp.status_code == 200:
            latencies.append((end - start) * 1000)
            
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f"\n--- Latency Report ---")
    print(f"Total Requests: {len(latencies)}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"p95 Latency: {p95_latency:.2f} ms")

if __name__ == "__main__":
    measure_latency()