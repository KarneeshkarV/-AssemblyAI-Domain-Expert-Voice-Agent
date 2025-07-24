from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)
import assemblyai as aai
import logging
import os
import queue
import threading
import time
from dotenv import load_dotenv
from typing import Type, Callable, Optional

load_dotenv()

api_key = os.getenv("ASSEMBLY_AI_API_KEY")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationalStreamer:
    """Enhanced streaming client for conversational interactions with callback support."""
    
    def __init__(self, on_transcript_callback: Optional[Callable[[str], None]] = None, pause_duration: float = 3.0):
        self.transcript_queue = queue.Queue()
        self.on_transcript_callback = on_transcript_callback
        self.client = None
        self.is_running = False
        
        # 3-second pause detection variables
        self.pause_duration = pause_duration
        self.accumulated_transcript = ""
        self.last_activity_time = 0
        self.pause_timer = None
        self.timer_lock = threading.Lock()
        
    def _on_pause_detected(self):
        """Called when a 3-second pause is detected."""
        with self.timer_lock:
            if self.accumulated_transcript.strip():
                transcript = self.accumulated_transcript.strip()
                logger.info(f"Pause detected - Processing transcript: {transcript}")
                
                self.transcript_queue.put(transcript)
                if self.on_transcript_callback:
                    self.on_transcript_callback(transcript)
                
                # Clear accumulated transcript
                self.accumulated_transcript = ""
            
            # Clear the timer
            self.pause_timer = None
    
    def _reset_pause_timer(self):
        """Reset the pause detection timer."""
        with self.timer_lock:
            # Cancel existing timer if any
            if self.pause_timer:
                self.pause_timer.cancel()
            
            # Start new timer for pause detection
            self.pause_timer = threading.Timer(self.pause_duration, self._on_pause_detected)
            self.pause_timer.start()
    
    def on_begin(self, client: StreamingClient, event: BeginEvent):
        logger.info(f"Session started: {event.id}")
        
    def on_turn(self, client: StreamingClient, event: TurnEvent):
        logger.info(f"Transcript: {event.transcript} (end_of_turn: {event.end_of_turn})")
        
        # Accumulate transcript parts
        if event.transcript:
            self.accumulated_transcript = event.transcript
            self.last_activity_time = time.time()
            
            # Reset the pause timer - we have new activity
            self._reset_pause_timer()
            
        # Keep turn formatting enabled
        if not event.turn_is_formatted:
            params = StreamingSessionParameters(format_turns=True)
            client.set_params(params)
                
    def on_terminated(self, client: StreamingClient, event: TerminationEvent):
        logger.info(f"Session terminated: {event.audio_duration_seconds} seconds of audio processed")
        self.is_running = False
        
        # Cancel any pending timer and process remaining transcript
        with self.timer_lock:
            if self.pause_timer:
                self.pause_timer.cancel()
                self.pause_timer = None
            
            # Process any remaining accumulated transcript
            if self.accumulated_transcript.strip():
                transcript = self.accumulated_transcript.strip()
                logger.info(f"Session ended - Processing remaining transcript: {transcript}")
                
                self.transcript_queue.put(transcript)
                if self.on_transcript_callback:
                    self.on_transcript_callback(transcript)
                
                self.accumulated_transcript = ""
        
    def on_error(self, client: StreamingClient, error: StreamingError):
        logger.error(f"Streaming error: {error}")
        
    def start_streaming(self):
        """Start the streaming session."""
        if not api_key:
            raise ValueError("ASSEMBLY_AI_API_KEY not found in environment")
            
        self.client = StreamingClient(
            StreamingClientOptions(
                api_key=api_key,
                api_host="streaming.assemblyai.com",
            )
        )
        
        self.client.on(StreamingEvents.Begin, self.on_begin)
        self.client.on(StreamingEvents.Turn, self.on_turn)
        self.client.on(StreamingEvents.Termination, self.on_terminated)
        self.client.on(StreamingEvents.Error, self.on_error)
        
        self.client.connect(
            StreamingParameters(
                sample_rate=16000,
                format_turns=True,
            )
        )
        
        self.is_running = True
        try:
            self.client.stream(aai.extras.MicrophoneStream(sample_rate=16000))
        except KeyboardInterrupt:
            logger.info("Streaming interrupted by user")
        finally:
            self.stop_streaming()
            
    def stop_streaming(self):
        """Stop the streaming session."""
        if self.client:
            self.client.disconnect(terminate=True)
            self.is_running = False
            
        # Cancel any pending timer
        with self.timer_lock:
            if self.pause_timer:
                self.pause_timer.cancel()
                self.pause_timer = None
            
    def get_transcript(self, timeout: Optional[float] = None) -> Optional[str]:
        """Get the next transcript from the queue."""
        try:
            return self.transcript_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# Legacy functions for backward compatibility
def on_begin(self: Type[StreamingClient], event: BeginEvent):
    print(f"Session started: {event.id}")


def on_turn(self: Type[StreamingClient], event: TurnEvent):
    print(f"{event.transcript} ({event.end_of_turn})")

    if event.end_of_turn and not event.turn_is_formatted:
        params = StreamingSessionParameters(
            format_turns=True,
        )

        self.set_params(params)


def on_terminated(self: Type[StreamingClient], event: TerminationEvent):
    print(
        f"Session terminated: {event.audio_duration_seconds} seconds of audio processed"
    )


def on_error(self: Type[StreamingClient], error: StreamingError):
    print(f"Error occurred: {error}")


def main():
    client = StreamingClient(
        StreamingClientOptions(
            api_key=api_key,
            api_host="streaming.assemblyai.com",
        )
    )

    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    client.connect(
        StreamingParameters(
            sample_rate=16000,
            format_turns=True,
        )
    )

    try:
        client.stream(aai.extras.MicrophoneStream(sample_rate=16000))
    finally:
        client.disconnect(terminate=True)


if __name__ == "__main__":
    main()
