from flask import Flask, request, jsonify
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

app = Flask(__name__)

processor = Wav2Vec2Processor.from_pretrained("indonesian-nlp/wav2vec2-large-xlsr-indonesian")
model = Wav2Vec2ForCTC.from_pretrained("indonesian-nlp/wav2vec2-large-xlsr-indonesian").to("cuda")
resampler = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)

@app.route("/stt", methods=["POST"])
def stt():
    file = request.files["file"]
    speech_array, sampling_rate = torchaudio.load(file)
    if sampling_rate != 16_000:
        speech_array = resampler(speech_array)
    inputs = processor(speech_array.squeeze().numpy(), sampling_rate=16_000, return_tensors="pt", padding=True)
    inputs = {key: tensor.to("cuda") for key, tensor in inputs.items()}
    with torch.no_grad():
        logits = model(inputs["input_values"], attention_mask=inputs["attention_mask"]).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(predicted_ids)[0]
    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(port=5001)
