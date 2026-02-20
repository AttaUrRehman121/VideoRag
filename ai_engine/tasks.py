"""
Utility helpers for working with transcripts (splitting into chunks).
Bubble will send us the transcript directly, so we don't download or
transcribe videos here anymore.
"""


def split_transcript(transcript: str, chunk_size: int = 80, chunk_overlap: int = 20) -> list:
    """
    Split transcript into chunks with word count and timing.
    
    Args:
        transcript: Full transcript text
        chunk_size: Number of words per chunk
        chunk_overlap: Number of words to overlap between chunks
        
    Returns:
        List of chunk dicts with text and timing info
    """
    words = transcript.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        # Estimate timing (rough approximation: ~150 words per minute)
        start_idx = i
        end_idx = min(i + chunk_size, len(words))
        
        # Rough time estimation based on word count
        start_time = int((start_idx / len(words)) * 3600) if words else 0  # Assume 1 hour video
        end_time = int((end_idx / len(words)) * 3600) if words else 0
        
        chunks.append({
            'text': chunk_text,
            'start_time': start_time,
            'end_time': end_time
        })
    
    return chunks
