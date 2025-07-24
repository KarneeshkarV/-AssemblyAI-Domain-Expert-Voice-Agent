from agent.analysis_engine import finance_agent, memory_agent_query
from assemblyAi.tts import get_transcript_from_youtube
from qdrant_rag.rag_client import RagClient
from agent.medical_analysis_engine import medical_analysis_team
# from file_saver.save import save_text_to_file

# from qdrant_rag.rag_client import RagClient

# transcript = get_transcript_from_file(
# "./sample_audio/12 Grim Rules for a Perfect Life.mp3"
# )

# rag_client = RagClient()
# rag_client.inject_text(transcript, name="12 Grim Rules for a Perfect Life")
# save_text_to_file(transcript, "./grim.txt")

# youtube_transcript = get_transcript_from_youtube(
# "https://www.youtube.com/watch?v=hMIc0my5U8g"
# )
# rag_client.inject_text(youtube_transcript, name="YouTube Video Transcript")
# save_text_to_file(youtube_transcript, "./youtube_transcript.txt")
# finance_agent(
# "what is the best investment strategy do a sweep market analysis and look at my personal docs for the best investment strategy and based on the current market suggest a portfolio which consists of nvidia , netflix , nasdaq , nifty 50 and P&G ,which can give me max profits with least risks and do store it in my memory?"
# )
memory_agent_query("Tell me all about the past memory")
# memory_agent_query(
# "Go in detail and teach me abuot retirement planning and how to do it, so that i can live good , with a small inflaction in lifestyle  do save this part to memory about my retierment and prefrences",
# debug=True,
# tui=True,
# )

medical_analysis_team(
    "how can i learn medin so that i do not need a docter as much as i need now ",
    user="1",
    debug=True,
    tui=True,
)
