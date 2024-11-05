import torch
import torchaudio
import os
import vlc
from openai import OpenAI
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Set backend to 'sox_io' to ensure MP3 compatibility
torchaudio.set_audio_backend("sox_io")

# Load model and processor
processor = Wav2Vec2Processor.from_pretrained(
    "indonesian-nlp/wav2vec2-large-xlsr-indonesian"
)
model = Wav2Vec2ForCTC.from_pretrained(
    "indonesian-nlp/wav2vec2-large-xlsr-indonesian"
).to(
    "cuda"
)  # Move model to GPU

# Resample audio to 16kHz
resampler = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)

# Load the specific MP3 file
file_path = "./cuaca_16.mp3"
speech_array, sampling_rate = torchaudio.load(file_path)

# Resample if the sampling rate is different from 16kHz
if sampling_rate != 16_000:
    speech_array = resampler(speech_array)

# Process the audio to prepare it for the model
inputs = processor(
    speech_array.squeeze().numpy(),
    sampling_rate=16_000,
    return_tensors="pt",
    padding=True,
)
inputs = {
    key: tensor.to("cuda") for key, tensor in inputs.items()
}  # Move input tensors to GPU

# Perform inference
with torch.no_grad():
    logits = model(
        inputs["input_values"], attention_mask=inputs["attention_mask"]
    ).logits

# Get predictions and decode them
predicted_ids = torch.argmax(logits, dim=-1)
pred_array = processor.batch_decode(predicted_ids)
print("Prediction:", pred_array)

# OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

stt_res = pred_array[0]

print("\n")
print(f"Speech to Text result: {stt_res}")

vlc.MediaPlayer(f"file://{os.getcwd()}/cuaca_16.mp3").play()


prompt_res = f"""This is the result from an external speech to text program without clear punctuations and might be jumbled. 
Please refine the output and answer to the question: {stt_res}"""

print("\n")
print(f"Prompt: {prompt_res}")

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
