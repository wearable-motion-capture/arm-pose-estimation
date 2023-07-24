"""
The microphone transcription thread will look for these keywords.
If it finds a keyword in its speech input, it will add the corresponding ID to the keywords queue.
See the transcription script for details.
"""
KEY_PHRASES = {
    # dual manipulation task
    "blue sky": 0,  # left
    "red apple": 1,  # right

    # ROS experiment commands
    "stop": 3,  # stop current task
    "continue": 4,  # carry on with original task
    "follow me": 5,  # move EE to hand
    "open": 6,  # open the gripper

    # simple test
    "test": 99
}
