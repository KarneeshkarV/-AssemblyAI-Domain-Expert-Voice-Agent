import os
import tempfile

import assemblyai as aai
import yt_dlp
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLY_AI_API_KEY")


def speed_up_audio_file(audio_file_path, output_dir=None, playback_speed=1.5):
    audio = AudioSegment.from_file(audio_file_path)
    sped_up_audio = audio.speedup(playback_speed=playback_speed)

    if output_dir is None:
        output_dir = os.path.dirname(audio_file_path)

    sped_up_path = os.path.join(output_dir, f"sped_up_{playback_speed}x_audio.wav")
    sped_up_audio.export(sped_up_path, format="wav")

    print(f"Speed up audio done ({playback_speed}x)")
    print(
        f"Original duration: {len(audio)/1000:.1f}s, Processed duration: {len(sped_up_audio)/1000:.1f}s"
    )

    return sped_up_path


def get_transcript_from_file(audio_file, speed_up=True, playback_speed=1.5):
    if speed_up:
        audio_file = speed_up_audio_file(audio_file, playback_speed=playback_speed)

    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.best
        # , auto_highlights=True, entity_detection=True
    )

    transcript = aai.Transcriber(config=config).transcribe(audio_file)

    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")
#     for result in transcript.auto_highlights.results:
        # print(
            # f"Highlight: {result.text}, Count: {result.count}, Rank: {result.rank}, Timestamps: {result.timestamps}"
        # )

    # for entity in transcript.entities:
        # print(entity.text)
        # print(entity.entity_type)
        # print(f"Timestamp: {entity.start} - {entity.end}\n")
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

        sped_up_path = speed_up_audio_file(audio_path, temp_dir)
        transcript = get_transcript_from_file(sped_up_path, speed_up=False)

        print(f"Successfully transcribed YouTube video: '{video_title}'")

        return transcript
