import json
from datetime import datetime
import subprocess
import ollama
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Global variables
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

available_models = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large-v3",
    "distil-whisper/distil-large-v2",
    "distil-whisper/distil-medium.en",
]

# Global pipeline variable
pipe = None


def load_model(model_id: str) -> None:
    global pipe
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    print(f"Speech recognition model '{model_id}' loaded successfully.")


def transcribe_audio(audio_path: str, model_id: str = "openai/whisper-small") -> str:
    global pipe
    if pipe is None or pipe.model.config._name_or_path != model_id:
        load_model(model_id)

    # Check if file format is not wav or mp3
    if audio_path.split(".")[-1] not in ["wav", "mp3"]:
        # Convert the audio file to mp3
        file_name = audio_path.split(".")[0]
        subprocess.run(["ffmpeg", "-i", audio_path, f"{file_name}.mp3"])
        audio_path = f"{file_name}.mp3"

    transcription = pipe(audio_path)
    return transcription["text"]


def summarize_with_ollama(text: str) -> dict:
    system_prompt = """You are an expert summarizer. Your task is to provide a concise summary of the given text. 
    The summary should capture the main points and key information.
    You must return your response in a JSON format with a suitable short title and a markdown summary.
    The JSON structure should look like this: 
    {
        "title": "A suitable title",
        "summary": "A markdown summary"
    }
    Ensure your response is a valid JSON object."""

    max_attempts = 3
    attempt = 0
    content_dict = None

    while attempt < max_attempts:
        try:
            response = ollama.chat(
                model="llama3.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Here is the transcription to summarize: {text}",
                    },
                ],
            )

            message = response["message"]
            content = message["content"]
            content_dict = json.loads(content)
            break
        except json.JSONDecodeError:
            print(f"Attempt {attempt + 1} failed to produce valid JSON. Retrying...")
            attempt += 1

    if content_dict is None:
        raise Exception("Failed to get a valid JSON response after 3 attempts.")

    return content_dict


def save_summary(summary: dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = f"./summaries/{timestamp+'_'+summary['title']}.md"
    with open(file_path, "w") as f:
        f.write(summary["summary"])
    return file_path


def process_audio(audio_path: str, model_id: str = "openai/whisper-small") -> dict:
    transcription = transcribe_audio(audio_path, model_id)
    summary = summarize_with_ollama(transcription)
    file_path = save_summary(summary)
    return {"transcription": transcription, "summary": summary, "file_path": file_path}
