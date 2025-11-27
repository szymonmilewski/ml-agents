import psutil
import csv
import time
from datetime import datetime

LOG_INTERVAL = 5
CSV_FILE = "resource_log.csv"

def init_csv(file):
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp",
            "CPU_Usage_percent",
            "RAM_Usage_percent",
            "RAM_Used_MB",
            "Disk_Read_MBps",
            "Disk_Write_MBps",
            "Net_Sent_KBps",
            "Net_Recv_KBps"
        ])

def log_resources(file):
    prev_disk = psutil.disk_io_counters()
    prev_net = psutil.net_io_counters()
    prev_time = time.time()

    while True:
        time.sleep(LOG_INTERVAL)
        now = time.time()

        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()

        disk = psutil.disk_io_counters()
        dt = now - prev_time
        read_rate = (disk.read_bytes - prev_disk.read_bytes) / dt / (1024 * 1024)
        write_rate = (disk.write_bytes - prev_disk.write_bytes) / dt / (1024 * 1024)

        net = psutil.net_io_counters()
        sent_rate = (net.bytes_sent - prev_net.bytes_sent) / dt / 1024
        recv_rate = (net.bytes_recv - prev_net.bytes_recv) / dt / 1024

        with open(file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                cpu,
                ram.percent,
                round(ram.used / (1024 * 1024), 2),
                round(read_rate, 2),
                round(write_rate, 2),
                round(sent_rate, 2),
                round(recv_rate, 2)
            ])

        prev_disk = disk
        prev_net = net
        prev_time = now


if __name__ == "__main__":
    init_csv(CSV_FILE)
    print(f"Logging system resources every {LOG_INTERVAL}s")
    log_resources(CSV_FILE)
