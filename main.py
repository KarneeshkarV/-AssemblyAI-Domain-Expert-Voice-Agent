from assemblyAi.tts import get_transcript_from_file
from rag.rag_ingest import inject_text


def main():
    transcript = get_transcript_from_file("https://assembly.ai/wildfires.mp3")
    print(transcript)
    inject_text(transcript, collection_name="Test1")
    print("Done")
