import sounddevice as sd
import numpy as np


def record_until_silence(
    samplerate=16000,
    silence_threshold=500,    # amplitude below this = silence
    silence_duration=3,     # seconds of silence before stopping
):
    input("Press Enter to start recording...")
    chunks = []
    silent_chunks = 0
    max_silent_chunks = int(silence_duration * samplerate / 1024)

    with sd.InputStream(samplerate=samplerate, channels=1, dtype=np.int16) as stream:
        print("Listening...")
        while True:
            chunk, _ = stream.read(1024)
            chunks.append(chunk)

            # check if chunk is silent
            volume = np.abs(chunk).mean()
            if volume < silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0   # reset on any speech

            # stop after N consecutive silent chunks
            if silent_chunks >= max_silent_chunks:
                print("Silence detected, stopping.")
                break

    return np.concatenate(chunks), samplerate