import platform,socket,re,uuid,json,psutil,logging, os, datetime

def getSystemInfo():
    try:
        info={}
        info['platform']=platform.system()
        info['platform-release']=platform.release()
        info['platform-version']=platform.version()
        info['architecture']=platform.machine()
        info['hostname']=socket.gethostname()
        info['ip-address']=socket.gethostbyname(socket.gethostname())
        info['mac-address']=':'.join(re.findall('..', '%012x' % uuid.getnode()))
        info['processor']=platform.processor()
        info['ram']=str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"
        info['cpu-count'] = os.cpu_count()
        info['boot-time'] = datetime.datetime.fromtimestamp(psutil.boot_time()).strftime("%Y-%m-%d %H:%M:%S")
        info['disk-partitions'] = [p.device for p in psutil.disk_partitions()]
        disk = psutil.disk_usage('/')
        info['disk-total'] = f"{disk.total / (1024**3):.2f} GB"
        info['disk-used'] = f"{disk.used / (1024**3):.2f} GB"
        info['disk-free'] = f"{disk.free / (1024**3):.2f} GB"
        info['disk-percent'] = f"{disk.percent}%"
        info['cpu-usage'] = f"{psutil.cpu_percent(interval=1)}%"


        return json.dumps(info, indent=4)
    except Exception as e:
        logging.exception(e)

print(getSystemInfo())

