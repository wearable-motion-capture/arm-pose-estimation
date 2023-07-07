import os

proj_path = os.path.dirname(os.path.abspath(__file__))

paths = {
    "deploy": "{}/data_deploy/".format(proj_path),
    "skeleton": "{}/data_deploy/".format(proj_path),
    "cache": "{}/cache".format(proj_path)
}

# your local IP
IP = "192.168.1.138"
# IP of machine running motive (in case you use mocap)
MOTIVE_SERVER = "192.168.1.116"
