import platform,socket,re,uuid,json,psutil,logging, os, datetime, GPUtil, subprocess


def get_pc_stats():
    pc_stats = {}

    pc_stats["cpu_type"] = platform.processor()
    pc_stats["cpu_cores"] = os.cpu_count()
    pc_stats["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3))
    pc_stats["os_name"] = platform.system()

    #dedicated NVIDIA GPU handling
    #TODO: add vram 
    gpus = GPUtil.getGPUs()

    pc_stats["has_nvidia"] = 1 if gpus else 0
    pc_stats["nvidia_gpu_name"] = gpus[0].name if gpus else "NO DEDICATED GPU"

    return pc_stats