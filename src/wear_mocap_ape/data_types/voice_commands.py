"""
The microphone transcription thread will look for these keywords.
If it finds a keyword in its speech input, it will add the corresponding ID to the keywords queue.
See the transcription script for details.
"""
KEY_PHRASES = {
    "stop": 3,  # stop current task
    # "move": 4,  # carry on with original task
    "continue": 4,  # same as "move"
    "now": 4,
    "follow": 5,  # move EE to hand
    "hand": 5,

    # "close": 7,

    # "handover": 6,  # same as "open"
    # "over": 6,
    "open": 6,  # open the gripper
    # "give": 6,
    # "cube": 6,
    "back": 10,
    "home": 10
}
