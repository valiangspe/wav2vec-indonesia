import requests
import vlc
import time
from g2p_id import G2P
from openai import OpenAI
import os

def play_audio_with_block(file_path):
    player = vlc.MediaPlayer(file_path)
    event_manager = player.event_manager()
    finished = False

    def on_end(event):
        nonlocal finished
        finished = True

    event_manager.event_attach(vlc.EventType.MediaPlayerEndReached, on_end)
    player.play()
    while not finished:
        time.sleep(0.1)


# Initialize G2P for text conversion
g2p = G2P()

# Play original audio
file_path = "./cuaca_16.mp3"

# Send audio to STT server
with open(file_path, "rb") as f:
    stt_response = requests.post("http://localhost:5001/stt", files={"file": f})

stt_text = stt_response.json().get("text")
print("STT Result:", stt_text)

print(f"Playing: {stt_text}")

play_audio_with_block(file_path)


prompt_res = f"""
This is the result from an external speech to text program without clear punctuations and might be jumbled. 
Please refine the output and answer to the question: {stt_text}

And you should only output the answer, as your result will be given to a text to speech machine.
"""

print("\n")
print(f"Prompt: {prompt_res}")

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


resp = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt_res,
        }
    ],
    model="gpt-4o-mini",
)

result = resp.choices[0].message.content

print("\n")
print(f"Result: {result}")


# Convert text with g2p
converted_text = g2p(result)

print("\n")
print("Converted Text for TTS (phoneme):", converted_text)

# Send converted text to TTS server
tts_url = f"http://localhost:5002/api/tts?text={converted_text}&speaker_id=ardi&style_wav=&language_id="
tts_response = requests.get(tts_url)

# Save the received MP3 file
tts_audio_path = "./tts-result.mp3"
with open(tts_audio_path, "wb") as f:
    f.write(tts_response.content)

# Play the TTS audio result
play_audio_with_block(tts_audio_path)
