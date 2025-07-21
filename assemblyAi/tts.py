import os

import assemblyai as aai
from dotenv import load_dotenv

load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLY_AI_API_KEY")


def get_transcript_from_file(audio_file):
    config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)

    transcript = aai.Transcriber(config=config).transcribe(audio_file)

    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")
    return transcript.text


if __name__ == "__main__":
    transcript = get_transcript("https://assembly.ai/wildfires.mp3")
    print(transcript)
