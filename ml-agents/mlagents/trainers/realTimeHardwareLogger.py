import psutil
import csv
import time
from datetime import datetime
import os
import sys
from pathlib import Path

LOG_INTERVAL = 5

now = datetime.now()
ram_csv_name = sys.argv[1]
CSV_FILE = ram_csv_name

def init_csv(file):
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp",
            "RAM_Usage_percent",
            "RAM_Avg_percent",
            "RAM_Used_MB"
        ])


def log_resources(file):
    prev_time = time.time()

    count = 0
    ram_sum = 0

    while True:
        time.sleep(LOG_INTERVAL)
        now = time.time()
        dt = now - prev_time

        ram_obj = psutil.virtual_memory()
        ram_pct = ram_obj.percent
        ram_used = ram_obj.used / (1024 * 1024)

        count += 1
        ram_sum += ram_pct

        ram_avg = ram_sum / count


        with open(file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round(ram_pct, 2),
                round(ram_avg, 2),
                round(ram_used, 2)
            ])

        prev_time = now

#FUNCTION: extracts maximum RAM usage %, average RAM usage % and maximum of RAM used in MB
def get_RAM_numbers(csv_file):
    max_ram_mb = 0
    max_ram_percent = 0
    avg_ram_percent = 0

    with open(csv_file, newline="", encoding="utf-8") as file: 
        reader = csv.DictReader(file)

        for row in reader:
            ram_mb = float(row["RAM_Used_MB"])
            max_ram_percent = float(row["RAM_Usage_percent"])
            avg_ram_percent = float(row["RAM_Avg_percent"])

            if ram_mb > max_ram_mb:
                max_ram_mb = ram_mb

    return max_ram_mb, max_ram_percent, avg_ram_percent

if __name__ == "__main__":
    if not os.path.isfile(CSV_FILE):
        init_csv(CSV_FILE)

    print(f"Logging system resources every {LOG_INTERVAL}s to {CSV_FILE}")
    log_resources(CSV_FILE)
