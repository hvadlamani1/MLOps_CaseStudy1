import gradio as gr
import torch
import torchaudio
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration


# --- Model Loading with Mac Fixes ---
def load_model():
    # Check for CUDA (Nvidia)
    if torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.float16
        print("Status: Detected Nvidia GPU. Using CUDA (float16).")

    # Check for MPS (Apple Silicon - M1/M2/M3)
    elif torch.backends.mps.is_available():
        device = "mps"
        # CRITICAL FIX: Macs must use float32.
        # float16 on MPS causes the "!!!!!!" infinite loop bug.
        torch_dtype = torch.float32
        print("Status: Detected Apple Silicon. Using MPS (float32) to prevent loops.")

    # Fallback to CPU
    else:
        device = "cpu"
        torch_dtype = torch.float32
        print("Status: No GPU detected. Using CPU (slow).")

    # Load model with the determined precision
    model = WhisperForConditionalGeneration.from_pretrained(
        "tclin/whisper-large-v3-turbo-atcosim-finetune",
        torch_dtype=torch_dtype
    )
    model = model.to(device)

    processor = WhisperProcessor.from_pretrained("tclin/whisper-large-v3-turbo-atcosim-finetune")

    return model, processor, device, torch_dtype


# Load resources once
model, processor, device, torch_dtype = load_model()


# --- Transcription Logic ---
def transcribe_audio(audio_file):
    if audio_file is None:
        return "Please upload an audio file"

    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_file)

        # Resample to 16kHz (Whisper requirement)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Convert Stereo to Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Prepare for model
        waveform_np = waveform.squeeze().cpu().numpy()

        input_features = processor(
            waveform_np,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features

        # Move to device (MPS/CUDA) with correct precision
        input_features = input_features.to(device=device, dtype=torch_dtype)

        # Generate transcription
        # Added repetition_penalty as an extra safety guard against loops
        generated_ids = model.generate(
            input_features,
            max_new_tokens=128,
            repetition_penalty=1.1
        )

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription

    except Exception as e:
        return f"Error processing audio: {str(e)}"


# --- Gradio Interface ---
demo = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="ATC Speech Transcription",
    description=f"Running on: {device.upper()} (Precision: {str(torch_dtype).split('.')[-1]})",
    examples=[
        ["atc-sample-1.wav"],
        ["atc-sample-2.wav"],
        ["atc-sample-3.wav"]
    ],
    article="This model is fine-tuned on the ATCOSIM dataset. If you see '!!!!!!' loops, ensure you are running in float32 mode."
)

if __name__ == "__main__":
    demo.launch()