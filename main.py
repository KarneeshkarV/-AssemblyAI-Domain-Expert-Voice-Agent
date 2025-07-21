import asyncio
from agent.analysis_engine import qdrant_agent
from assemblyAi.tts import get_transcript_from_file
from rag.rag_client import RagClient

# transcript = get_transcript_from_file("./sample_audio/12 Grim Rules for a Perfect Life.mp3")
# rag_client = RagClient()
# rag_client.inject_text(transcript)

asyncio.run(knowledge_base.aload(recreate=False))
qdrant_agent("where are the smokes from ?")
