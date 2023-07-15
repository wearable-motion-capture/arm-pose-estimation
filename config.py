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

# ports for streaming to unity
UNITY_MOTIVE_PORT = 50005

UNITY_WATCH_PHONE_PORT_LEFT = 50003
UNITY_WATCH_PHONE_PORT_RIGHT = 50004
UNITY_WATCH_PORT = 50003

# Listener ports for receiving data
WATCH_PHONE_PORT_LEFT = 65000
WATCH_PHONE_PORT_RIGHT = 65003
WATCH_PORT = 46000
