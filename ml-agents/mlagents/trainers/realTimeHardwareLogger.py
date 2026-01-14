import psutil
import csv
import time
from datetime import datetime
import os

LOG_INTERVAL = 5

now = datetime.now()
CSV_FILE = "resource_log_" + now.strftime('%Y-%m-%d_%H-%M-%S') + ".csv"


def init_csv(file):
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp",
            "CPU_Usage_percent",
            "CPU_Avg_percent",
            "RAM_Usage_percent",
            "RAM_Avg_percent",
            "RAM_Used_MB",
            "Disk_Read_MBps",
            "Disk_Write_MBps",
            "Disk_Read_Avg_MBps",
            "Disk_Write_Avg_MBps",
            "Net_Sent_KBps",
            "Net_Recv_KBps",
            "Net_Sent_Avg_KBps",
            "Net_Recv_Avg_KBps"
        ])


def log_resources(file):
    prev_disk = psutil.disk_io_counters()
    prev_net = psutil.net_io_counters()
    prev_time = time.time()

    count = 0
    cpu_sum = 0
    ram_sum = 0
    disk_read_sum = 0
    disk_write_sum = 0
    net_sent_sum = 0
    net_recv_sum = 0

    while True:
        time.sleep(LOG_INTERVAL)
        now = time.time()
        dt = now - prev_time

        cpu = psutil.cpu_percent(interval=None)
        ram_obj = psutil.virtual_memory()
        ram_pct = ram_obj.percent
        ram_used = ram_obj.used / (1024 * 1024)

        disk = psutil.disk_io_counters()
        read_rate = (disk.read_bytes - prev_disk.read_bytes) / dt / (1024 * 1024)
        write_rate = (disk.write_bytes - prev_disk.write_bytes) / dt / (1024 * 1024)

        net = psutil.net_io_counters()
        sent_rate = (net.bytes_sent - prev_net.bytes_sent) / dt / 1024
        recv_rate = (net.bytes_recv - prev_net.bytes_recv) / dt / 1024

        count += 1
        cpu_sum += cpu
        ram_sum += ram_pct
        disk_read_sum += read_rate
        disk_write_sum += write_rate
        net_sent_sum += sent_rate
        net_recv_sum += recv_rate

        cpu_avg = cpu_sum / count
        ram_avg = ram_sum / count
        disk_read_avg = disk_read_sum / count
        disk_write_avg = disk_write_sum / count
        net_sent_avg = net_sent_sum / count
        net_recv_avg = net_recv_sum / count

        with open(file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round(cpu, 2),
                round(cpu_avg, 2),
                round(ram_pct, 2),
                round(ram_avg, 2),
                round(ram_used, 2),
                round(read_rate, 2),
                round(write_rate, 2),
                round(disk_read_avg, 2),
                round(disk_write_avg, 2),
                round(sent_rate, 2),
                round(recv_rate, 2),
                round(net_sent_avg, 2),
                round(net_recv_avg, 2)
            ])

        prev_disk = disk
        prev_net = net
        prev_time = now

def log_hw():
    if not os.path.isfile(CSV_FILE):
        init_csv(CSV_FILE)

    print(f"Logging system resources every {LOG_INTERVAL}s to {CSV_FILE}")
    log_resources(CSV_FILE)

if __name__ == "__main__":
    if not os.path.isfile(CSV_FILE):
        init_csv(CSV_FILE)

    print(f"Logging system resources every {LOG_INTERVAL}s to {CSV_FILE}")
    log_resources(CSV_FILE)
