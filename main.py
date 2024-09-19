import json
from datetime import datetime
import subprocess
import ollama
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "openai/whisper-large-v3"
model_id = "distil-whisper/distil-large-v3"

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

# Load the audio data
audio_file_path = "data/1min_norsk_engelsk_test.mp3"

# Check if file format is not wav or mp3
if audio_file_path.split(".")[-1] not in ["wav", "mp3"]:
    # Convert the audio file to mp3
    # Get the file name without the extension
    file_name = audio_file_path.split(".")[0]
    # Convert the file to mp3
    subprocess.run(["ffmpeg", "-i", audio_file_path, f"{file_name}.mp3"])
    # Update the audio file path
    audio_file_path = f"{file_name}.mp3"

# Transcribe the audio
transcription = pipe(audio_file_path)

# Print the transcription
print(transcription)

# Use local ollama model to summarize the transcription
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
        response = ollama.chat(model='llama3.1', messages=[
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': f'Here is the transcription to summarize: {transcription["text"]}',
            },
        ])

        print(f"Response: {response}")

        # Get the message content from the response
        message = response['message']
        content = message['content']
        content_dict = json.loads(content)
        print(f"Content_dict: {content_dict} +  type: {type(content_dict)}")

        # If we successfully parse the JSON, break the loop
        break
    except json.JSONDecodeError:
        print(f"Attempt {attempt + 1} failed to produce valid JSON. Retrying...")
        attempt += 1

if content_dict is None:
    print("Failed to get a valid JSON response after 3 attempts.")
else:
    title = content_dict['title']
    summary = content_dict['summary']

    # Save the content to a markdown file
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = f"./summaries/{timestamp+'_'+title}.md"
    with open(file_path, "w") as f:
        f.write(summary)

    print(f"Summary saved to {file_path}")