import platform,socket,re,uuid,json,psutil,logging, os, datetime, GPUtil, subprocess
import cpuinfo 

def get_pc_stats():
    pc_stats = {}

    pc_stats["cpu_type"] = cpuinfo.get_cpu_info()["brand_raw"]
    pc_stats["cpu_freq"] = round(cpuinfo.get_cpu_info()["hz_advertised"][0] / 1e9, 2)
    pc_stats["cpu_cores"] = os.cpu_count()
    pc_stats["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3))
    pc_stats["os_name"] = platform.system()

    #dedicated NVIDIA GPU handling
    gpus = GPUtil.getGPUs()

    pc_stats["has_nvidia"] = 1 if gpus else 0
    pc_stats["nvidia_gpu_name"] = gpus[0].name if gpus else "NO DEDICATED GPU"
    pc_stats["nvidia_vram_gb"] = round(gpus[0].memoryTotal / 1024, 2) if gpus else 0.0

    return pc_stats