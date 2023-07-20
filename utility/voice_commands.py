"""
The microphone transcription thread will look for these keywords.
If it finds a keyword in its speech input, it will add the corresponding ID to the keywords queue
"""
KEY_PHRASES = {
    "blue sky": 0,
    "red apple": 1,
    "hello": 98,
    "test": 99
}
