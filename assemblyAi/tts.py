import os
import tempfile

import assemblyai as aai
import yt_dlp
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLY_AI_API_KEY")


def get_transcript_from_file(audio_file):
    config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)

    transcript = aai.Transcriber(config=config).transcribe(audio_file)

    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")
    return transcript.text


def get_transcript_from_youtube(youtube_url):
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": f"{temp_dir}/%(title)s.%(ext)s",
            "extractaudio": True,
            "audioformat": "wav",
            "audioquality": 1,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            video_title = info.get("title", "unknown")

            for filename in os.listdir(temp_dir):
                if filename.endswith((".wav", ".mp3", ".m4a", ".webm")):
                    audio_path = os.path.join(temp_dir, filename)
                    break
            else:
                raise RuntimeError("No audio file found after download")

        audio = AudioSegment.from_file(audio_path)
        sped_up_audio = audio.speedup(playback_speed=1.5)
        print("Speed up audio done")

        sped_up_path = os.path.join(temp_dir, "sped_up_audio.wav")
        sped_up_audio.export(sped_up_path, format="wav")

        transcript = get_transcript_from_file(sped_up_path)

        print(f"Successfully transcribed YouTube video: '{video_title}'")
        print(
            f"Original duration: {len(audio)/1000:.1f}s, Processed duration: {len(sped_up_audio)/1000:.1f}s"
        )

        return transcript



