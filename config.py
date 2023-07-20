import os

proj_path = os.path.dirname(os.path.abspath(__file__))

paths = {
    "deploy": f"{proj_path}/data_deploy/",
    "skeleton": f"{proj_path}/data_deploy/",
    "cache": f"{proj_path}/cache"
}

# your local IP
IP = "192.168.1.138"

# IP of machine running motive (in case you use mocap)
MOTIVE_SERVER = "192.168.1.116"

# ports for publishing to other services
PUB_MOTIVE = 50005
PUB_WATCH_PHONE_LEFT = 50003
PUB_WATCH_PHONE_RIGHT = 50004
PUB_WATCH_IMU_LEFT = 50003
PUB_TRANSCRIBED_KEYS = 50006

# Listener ports for receiving data
LISTEN_WATCH_PHONE_IMU_LEFT = 65000
LISTEN_WATCH_PHONE_IMU_RIGHT = 65003
LISTEN_WATCH_PHONE_AUDIO = 65001
LISTEN_WATCH_IMU_LEFT = 46000
