import gradio as gr
import torch
import torchaudio
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import os

import soundfile as sf
import librosa


# --- Model Loading with Mac Fixes ---

# Global placeholders
model = None
processor = None
atc_translator = None

def detect_device():
    # Check for CUDA (Nvidia)
    if torch.cuda.is_available():
        print("Status: Detected Nvidia GPU. Using CUDA (float16).")
        return "cuda:0", torch.float16

    # Check for MPS (Apple Silicon - M1/M2/M3)
    elif torch.backends.mps.is_available():
        print("Status: Detected Apple Silicon. Using MPS (float32) to prevent loops.")
        return "mps", torch.float32

    # Fallback to CPU
    else:
        print("Status: No GPU detected. Using CPU (slow).")
        return "cpu", torch.float32

# Detect device immediately for Gradio interface labels
device, torch_dtype = detect_device()

def load_resources():
    global model, processor, atc_translator, device, torch_dtype
    
    if model is None:
        print("Loading Whisper model...")
        loaded_model = WhisperForConditionalGeneration.from_pretrained(
            "tclin/whisper-large-v3-turbo-atcosim-finetune",
            torch_dtype=torch_dtype
        )
        model = loaded_model.to(device)
        processor = WhisperProcessor.from_pretrained("tclin/whisper-large-v3-turbo-atcosim-finetune")

    if atc_translator is None:
        print("Loading Translator model...")
        # Qwen2.5-1.5B is highly compatible and very smart for its size
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto", 
            device_map="auto"
        )
        
        atc_translator = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)



def atc_english_translation(atc_prompt):
    load_resources()
    if not atc_prompt or "Error" in atc_prompt:
        return "Waiting for valid transcription..."

    # Qwen uses a standard chat template
    messages = [
        {"role": "system", "content": "You are an aviation expert. Translate the following technical ATC radio transmission into simple, conversational plain English. Do not give definitions, just simply translate to conversational english! Be concise."},
        {"role": "user", "content": atc_prompt}
    ]
    
    # We use the tokenizer's chat template for the best results
    prompt = atc_translator.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    try:
        outputs = atc_translator(
            prompt,
            do_sample=False,
            max_new_tokens=256,
            return_full_text=False
        )
        return outputs[0]['generated_text'].strip()
    except Exception as e:
        return f"Translation Error: {str(e)}"

# --- Transcription Logic ---
def transcribe_audio(audio_file):
    load_resources()
    if audio_file is None:
        return "Please upload an audio file"

    try:
        # 1. Use soundfile to load. It bypasses the libtorchcodec system errors.
        speech, sample_rate = sf.read(audio_file)
        
        # 2. Ensure it's float32 (Whisper requirement)
        speech = speech.astype(np.float32)

        # 3. Convert Stereo to Mono
        if len(speech.shape) > 1:
            speech = np.mean(speech, axis=1)

        # 4. Resample to 16kHz using librosa (more reliable than torchaudio on clusters)
        if sample_rate != 16000:
            speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)

        # 5. Prepare for Whisper
        input_features = processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features

        # Move to device (MPS/CUDA/CPU)
        input_features = input_features.to(device=device, dtype=torch_dtype)

        # 6. Generate transcription
        generated_ids = model.generate(
            input_features,
            max_new_tokens=128,
            repetition_penalty=1.1
        )

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        translated_text = atc_english_translation(transcription)

        return transcription, translated_text

    except Exception as e:
        return f"Error processing audio: {str(e)}"


# --- Gradio Interface ---
demo = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath"),
    outputs=[
        gr.Textbox(label="Step 1: Raw ATC Transcription"),
        gr.Textbox(label="Step 2: Plain English Interpretation")
    ],
    title="ATC Speech Transcription",
    description=f"Running on: {device.upper()}",
    
    # 1. REMOVE the examples for now 
    # (We will add them back once the base app works)
    examples=None, 
    
    # 2. Disable caching and flagging (common crash points in v4.44)
    cache_examples=False,
    allow_flagging="never"
)



if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_api=False)