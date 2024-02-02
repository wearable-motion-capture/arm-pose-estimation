from pathlib import Path

proj_path = Path(__file__).parent.absolute()

PATHS = {
    "deploy": proj_path / "data_deploy",
    "skeleton": proj_path / "data_deploy"
}

# ports for publishing to other services
PORT_PUB_LEFT_ARM = 50003

# Listener ports for receiving data
PORT_LISTEN_WATCH_PHONE_IMU_LEFT = 65000  # watch and phone either mode
PORT_LISTEN_WATCH_IMU_LEFT = 46000  # watch only

# experimental modes
PORT_LISTEN_AUDIO = 65001
PORT_PUB_TRANSCRIBED_KEYS = 50006

# PORT_LISTEN_WATCH_PHONE_IMU_RIGHT = 65003
# PORT_PUB_RIGHT_ARM = 50004
