import csv
import os

log_file_path = 'result_bw.csv'
total_bytes = 0

try:
    with open(log_file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row: # 确保行不为空
                total_bytes += int(row[0])

    total_kilobytes = total_bytes / 1024
    
    print("--- Bandwidth Usage Results (Original Method) ---")
    print(f"Total Bandwidth (from result_bw.csv): {total_kilobytes:.2f} KB")
    print("Note: This value primarily reflects the size of low-confidence cropped images and does not include high-confidence JSON data or HTTP overhead.")
    
    # (可选) 进行归一化处理
    data_dir = 'CameraSys/dataset/trafficcam_2/src'
    try:
        original_size = sum(os.path.getsize(os.path.join(data_dir, f)) for f in os.listdir(data_dir))
        original_kb = original_size / 1024
        
        print(f"\nOriginal Full-Frame Data Size: {original_kb:.2f} KB")
        if original_size > 0:
            print(f"Measured Bandwidth as % of Original: {total_bytes / original_size:.2%}")
    except FileNotFoundError:
        print(f"\nWarning: Could not find original data directory at '{data_dir}' to perform normalization.")

except FileNotFoundError:
    print(f"Error: Log file not found at '{log_file_path}'")
except Exception as e:
    print(f"An error occurred: {e}")
