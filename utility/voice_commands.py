"""
The microphone transcription thread will look for these keywords.
If it finds a keyword in its speech input, it will add the corresponding ID to the keywords queue
"""
commands = {
    "hello": 98,
    "test": 99
}
