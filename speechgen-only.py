from g2p_id import G2P
import subprocess
import time
import os
import vlc


# Define VLC player function with event handling
def play_audio_with_block(player):
    event_manager = player.event_manager()
    finished = False

    def on_end(event):
        nonlocal finished
        finished = True

    event_manager.event_attach(vlc.EventType.MediaPlayerEndReached, on_end)
    player.play()

    # Wait for playback to finish
    while not finished:
        time.sleep(0.1)


ori_text = "Apa yang dimaksud dengan I O T atau internet of things?"

print(f"Original: {ori_text}")

g2p = G2P()
text = g2p(ori_text)

print(f"Result g2p: {text}")

tts_command = f"""
docker run --rm --gpus all \
    -v ./speechgen-res:/root/tts-output \
    -v ./tts-config:/root/tts-config \
    -v ./speakers.pth:/root/speakers.pth \
    ghcr.io/coqui-ai/tts \
    --text "{text}" \
    --model_path /root/tts-config/checkpoint_1260000-inference.pth \
    --config_path /root/tts-config/config.json \
    --speaker_idx ardi \
    --out_path /root/tts-output/output.wav \
    --use_cuda true
"""

print("TTS command:")
print(tts_command)

subprocess.run(
    tts_command,
    shell=True,
    cwd=".",
)


print("Speaking:")
print(f"{ori_text}")


player_tts = vlc.MediaPlayer(f"file://{os.getcwd()}/speechgen-res/output.wav")
play_audio_with_block(player_tts)
