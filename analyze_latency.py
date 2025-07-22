# 创建一个新脚本，例如 analyze_latency.py

import numpy as np

log_file = 'ServerSys/backend/latency_log.txt' # 确认路径正确
latencies = []

try:
    with open(log_file, 'r') as f:
        for line in f:
            try:
                # 将延迟转换为毫秒
                latencies.append(float(line.strip()) * 1000)
            except ValueError:
                continue # 忽略空行或无效行

    if not latencies:
        print("Log file is empty or contains no valid data.")
    else:
        latencies_np = np.array(latencies)
        
        print("--- End-to-End Latency Results ---")
        print(f"Total data points: {len(latencies_np)}")
        print(f"Average Latency:   {np.mean(latencies_np):.2f} ms")
        print(f"Median Latency:    {np.median(latencies_np):.2f} ms")
        print(f"Min Latency:       {np.min(latencies_np):.2f} ms")
        print(f"Max Latency:       {np.max(latencies_np):.2f} ms")
        print(f"95th Percentile:   {np.percentile(latencies_np, 95):.2f} ms")
        print("------------------------------------")

except FileNotFoundError:
    print(f"Error: Log file not found at '{log_file}'")
except Exception as e:
    print(f"An error occurred: {e}")
