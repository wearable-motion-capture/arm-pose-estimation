from enum import Enum


class FF(Enum):
    WATCH_ONLY = "2b4e48b366d717b035751c40f977d9ae6c26d6b2"
    WATCH_POCKET_PHONE = "563c9b045296cb4ffe906742a2837471bb61d382"


class LSTM(Enum):
    WATCH_ONLY = "2c700a1ca1af084eedbae5bdd86a5194e42ded4d"
    POCKET_MODE = "ea7d49ddfc25408761b055ea3229ec81b29a1b07"
    IMU_POSE = "dd0812a5d3ac7aaedebd6ab77717ab31212c7e50"
    IMU_POSE_CAL = "63fc7c8abbd2fc94ab7492d7e8ea5b88cef399e6"
