import platform,socket,re,uuid,json,psutil,logging, os, datetime, GPUtil, subprocess
import cpuinfo

def get_pc_stats():
    info = cpuinfo.get_cpu_info()
    pc_stats = {}

    pc_stats["cpu_type"] = info.get("brand_raw", "unknown")
    pc_stats["cpu_cores"] = psutil.cpu_count(logical=False) or "unknown"

    # CPU frequency (macOS safe)
    pc_stats["cpu_freq"] = (
        info.get("hz_advertised_friendly")
        or info.get("hz_actual_friendly")
        or "unknown"
    )

    pc_stats["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    pc_stats["os_name"] = platform.system()

    # GPU fields — ALWAYS present
    pc_stats["has_nvidia"] = False
    pc_stats["nvidia_gpu_name"] = None
    pc_stats["nvidia_vram_gb"] = None

    return pc_stats
