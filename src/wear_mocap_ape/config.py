from pathlib import Path

proj_path = Path(__file__).parent.absolute()

PATHS = {
    "deploy": proj_path / "data_deploy",
    "skeleton": proj_path / "data_deploy"
}

# ports for publishing to other services
PORT_PUB_LEFT_ARM = 50003
PORT_PUB_RIGHT_ARM = 50004


# Listener ports for receiving data
# watch and phone
PORT_LISTEN_WATCH_PHONE_IMU_LEFT = 65000
PORT_LISTEN_WATCH_PHONE_IMU_RIGHT = 65003
PORT_LISTEN_WATCH_PHONE_AUDIO = 65001
# watch only
PORT_LISTEN_WATCH_AUDIO = 46001
PORT_LISTEN_WATCH_IMU_LEFT = 46000

# experimental modes
PORT_PUB_TRANSCRIBED_KEYS = 50006
