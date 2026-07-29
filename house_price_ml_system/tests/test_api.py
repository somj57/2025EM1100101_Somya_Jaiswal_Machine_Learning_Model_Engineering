import requests
import time
import numpy as np

def measure_latency():
    url = "http://localhost:8000/predict"
    payload = {
        "YrSold": 2023, "YearBuilt": 2000, "YearRemodAdd": 2010,
        "TotalBsmtSF": 1000, "1stFlrSF": 1000, "2ndFlrSF": 800,
        "FullBath": 2, "HalfBath": 1, "BsmtFullBath": 0,
        "BsmtHalfBath": 0, "PoolArea": 0, "OverallQual": 7
    }
    
    latencies = []
    for _ in range(100): # Send 100 requests
        start = time.time()
        res = requests.post(url, json=payload)
        latencies.append(time.time() - start)
        
    avg_latency = np.mean(latencies) * 1000
    p95_latency = np.percentile(latencies, 95) * 1000
    
    print(f"Total Requests: 100")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"P95 Latency: {p95_latency:.2f} ms")

if __name__ == "__main__":
    measure_latency()