import gradio as gr
import torch
import torchaudio
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Model loading function with caching
def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = WhisperForConditionalGeneration.from_pretrained("tclin/whisper-large-v3-turbo-atcosim-finetune")
    model = model.to(device=device, dtype=torch_dtype)
    processor = WhisperProcessor.from_pretrained("tclin/whisper-large-v3-turbo-atcosim-finetune")
    
    return model, processor, device, torch_dtype

# Load model and processor once at startup
model, processor, device, torch_dtype = load_model()

# Define the transcription function
def transcribe_audio(audio_file):
    # Check if audio file exists
    if audio_file is None:
        return "Please upload an audio file"
    
    try:
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(audio_file)
        
        # Resample to 16kHz (required for Whisper models)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # Convert to numpy array
        waveform_np = waveform.squeeze().cpu().numpy()
        
        # Process with model
        input_features = processor(waveform_np, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(device=device, dtype=torch_dtype)
        
        generated_ids = model.generate(input_features, max_new_tokens=128)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return transcription
        
    except Exception as e:
        return f"Error processing audio: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="ATC Speech Transcription",
    description="Convert Air Traffic Control (ATC) radio communications to text. Upload your own ATC audio or try the examples below.",
    examples=[
        ["atc-sample-1.wav"],
        ["atc-sample-2.wav"],
        ["atc-sample-3.wav"]
    ],
    article="This model is fine-tuned on the ATCOSIM dataset with a 3.73% Word Error Rate on ATC communications. It is specifically optimized for aviation terminology, callsigns, and standard phraseology. Audio should be 16kHz sample rate for best results."
)

# Launch the interface
if __name__ == "__main__":
    demo.launch()