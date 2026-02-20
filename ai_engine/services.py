"""
Core AI services for embeddings, vector search, and LLM responses.
Optimized for 2-3 second response times.
"""
import os
import json
import redis
import time
import threading
from datetime import datetime, timedelta
from django.conf import settings
from openai import OpenAI
from .models import TranscriptChunk, QuestionCache
from .tasks import split_transcript
from .utils import calculate_similarity

# Pinecone client (optional, for vector search testing)
try:
    from pinecone import Pinecone, ServerlessSpec
    _pinecone_available = True
except ImportError:
    _pinecone_available = False


# Initialize OpenAI client lazily to ensure settings are loaded
def get_openai_client():
    """Get OpenAI client, creating it if needed."""
    if not hasattr(get_openai_client, '_client'):
        if not settings.OPENAI_API_KEY:
            raise ValueError('OPENAI_API_KEY is not set in settings. Please set it in .env file and restart the server.')
        get_openai_client._client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=15.0,  # Reduced timeout for faster failures
            max_retries=1  # Single retry for speed
        )
    return get_openai_client._client

# For backward compatibility, create client at module load
try:
    openai_client = get_openai_client()
except ValueError:
    # Will be recreated when first used if .env is loaded later
    openai_client = None

# Initialize Redis client (optional, won't fail if Redis is unavailable)
try:
    redis_client = redis.from_url(settings.REDIS_URL)
except Exception:
    redis_client = None

# Initialize Pinecone client (optional, for vector search testing)
_pinecone_client = None
_pinecone_index = None
def get_pinecone_index():
    """Get Pinecone index, initializing if needed."""
    global _pinecone_client, _pinecone_index
    if not _pinecone_available:
        raise Exception('Pinecone package not installed. Run: pip install pinecone')
    if not settings.PINECONE_API_KEY:
        raise Exception('PINECONE_API_KEY not set in .env file')
    if not settings.PINECONE_INDEX_NAME:
        raise Exception('PINECONE_INDEX_NAME not set in .env file')
    if _pinecone_index is None:
        try:
            _pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
            _pinecone_index = _pinecone_client.Index(settings.PINECONE_INDEX_NAME)
        except Exception as e:
            raise Exception(f'Failed to connect to Pinecone: {str(e)}. Check your API key and index name.')
    return _pinecone_index


def generate_embedding(text: str) -> list:
    """
    Generate embedding for given text using OpenAI API.
    Optimized for speed with minimal timeout and retries.
    """
    # Ensure client is initialized (handles case where server started before .env was created)
    client = get_openai_client()
    
    if not settings.OPENAI_API_KEY:
        raise Exception('OPENAI_API_KEY is not configured. Please set it in .env file and restart the server.')
    
    try:
        # Use fastest embedding model with minimal timeout
        response = client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=text,
            timeout=10.0  # 10s timeout (was 30s)
        )
        return response.data[0].embedding
    except Exception as e:
        error_msg = str(e)
        # Provide more helpful error messages
        if 'Connection' in error_msg or 'timeout' in error_msg.lower():
            raise Exception(
                f'Failed to connect to OpenAI API: {error_msg}. '
                'Check your internet connection and API key validity. If you just created .env, restart the Django server.'
            )
        elif 'Invalid API key' in error_msg or '401' in error_msg:
            raise Exception(
                f'Invalid OpenAI API key. Please check OPENAI_API_KEY in .env file.'
            )
        else:
            raise Exception(f'Failed to generate embedding: {error_msg}')


def vector_search(question_embedding: list, video_id: str, top_k: int = 3) -> list:
    """
    Perform vector similarity search (SQLite: load chunks, rank by cosine similarity in Python).
    """
    try:
        chunks = list(
            TranscriptChunk.objects.filter(video_id=video_id).only(
                'id', 'text', 'start_time', 'end_time', 'embedding'
            )
        )
        if not chunks:
            return []
        # Rank by cosine similarity (higher = better)
        scored = []
        for c in chunks:
            emb = json.loads(c.embedding) if isinstance(c.embedding, str) else c.embedding
            sim = calculate_similarity(question_embedding, emb)
            scored.append((sim, c))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:top_k]]
    except Exception as e:
        raise Exception(f'Vector search failed: {str(e)}')


def format_timestamp(seconds: int) -> str:
    """Convert seconds to MM:SS format."""
    minutes = seconds // 60
    secs = seconds % 60
    return f'{minutes:02d}:{secs:02d}'


def build_context(chunks: list) -> str:
    """
    Build context from vector-retrieved chunks. Pass FULL chunk text - no truncation.
    Vector search already returns the most relevant chunks; LLM needs complete context.
    """
    import re
    context_parts = []
    for chunk in chunks:
        text = chunk.text.strip()
        # Remove timestamp noise (e.g. 00:00, 01:19) for readability only
        text = re.sub(r'\d{2}:\d{2}\s*', '', text)
        text = re.sub(r'\s+', ' ', text)
        context_parts.append(text.strip())
    return '\n\n---\n\n'.join(context_parts)  # Full context, clearly separated


def get_llm_response(question: str, context: str) -> str:
    """
    Get response from OpenAI GPT model using RAG context.
    Improved prompt to ensure LLM uses the provided context.
    """
    # Stronger system prompt that explicitly requires using context
    system = """You are a course assistant. Answer questions based ONLY on the provided context.
- If the context contains the answer, provide it clearly.
- If the context doesn't contain relevant information, say "The context does not specify this information."
- Be concise (max 3 short paragraphs, under 120 words)."""

    # Clear user prompt with explicit context section
    user_content = f"""Context from video transcript:
{context}

Question: {question}

Answer based on the context above:"""

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user_content}
            ],
            temperature=0.1,
            max_tokens=150,  # Increased from 60 to allow complete answers
            stream=False,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        if 'Connection' in error_msg or 'timeout' in error_msg.lower():
            raise Exception(
                f'Failed to connect to OpenAI API: {error_msg}. '
                'Check your internet connection and API key validity. If you just created .env, restart the Django server.'
            )
        raise Exception(f'LLM request failed: {error_msg}')


def get_llm_response_stream(question: str, context: str):
    """
    Stream response from OpenAI GPT using RAG context.
    Yields text chunks as they arrive. Caller can collect for caching.
    """
    system = """You are a course assistant. Answer questions based ONLY on the provided context.
- If the context contains the answer, provide it clearly.
- If the context doesn't contain relevant information, say "The context does not specify this information."
- Be concise (max 3 short paragraphs, under 120 words)."""
    user_content = f"""Context from video transcript:
{context}

Question: {question}

Answer based on the context above:"""
    try:
        client = get_openai_client()
        stream = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user_content}
            ],
            temperature=0.1,
            max_tokens=150,
            stream=True,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if getattr(delta, 'content', None):
                    yield delta.content
    except Exception as e:
        error_msg = str(e)
        if 'Connection' in error_msg or 'timeout' in error_msg.lower():
            raise Exception(
                f'Failed to connect to OpenAI API: {error_msg}. '
                'Check your internet connection and API key validity.'
            )
        raise Exception(f'LLM stream failed: {error_msg}')


def check_cache(video_id: str, question: str, similarity_threshold: float = 0.90, question_embedding: list | None = None) -> dict | None:
    """
    Check if similar question is cached.
    Pass question_embedding when already computed to avoid a second embedding API call.

    Args:
        video_id: Video ID to check cache for
        question: Question text
        similarity_threshold: Threshold for considering questions similar
        question_embedding: Optional precomputed embedding (avoids extra API call)

    Returns:
        Cached answer dict or None if not found
    """
    try:
        if question_embedding is None:
            question_embedding = generate_embedding(question)

        # Load all cached questions for this video, rank by similarity (SQLite-compatible)
        candidates = list(
            QuestionCache.objects.filter(video_id=video_id).prefetch_related('source_chunks')
        )
        best = None
        best_sim = similarity_threshold
        for c in candidates:
            emb = json.loads(c.question_embedding) if isinstance(c.question_embedding, str) else c.question_embedding
            sim = calculate_similarity(question_embedding, emb)
            if sim >= best_sim:
                best_sim = sim
                best = c
        cached = best
        
        if cached:
            # Skip access count update for speed (can be done async later if needed)
            
            # Use prefetched chunks (no extra query)
            return {
                'answer': cached.answer,
                'sources': [
                    {
                        'text': chunk.text,
                        'timestamp': format_timestamp(chunk.start_time),
                        'start_time': chunk.start_time,
                        'end_time': chunk.end_time
                    }
                    for chunk in cached.source_chunks.all()
                ],
                'cached': True
            }
    except Exception as e:
        # Log but don't fail on cache lookup
        print(f'[v0] Cache lookup error: {str(e)}')
    
    return None


def ask_ai_service(user, video_id: str, question: str, session_id: int = None) -> dict:
    """
    Main service for Ask AI functionality.
    
    Args:
        user: User object
        video_id: Video ID to ask about
        question: Question text
        session_id: Optional session ID
        
    Returns:
        Response dict with answer, sources, and metadata
    """
    # Fast-path for trivial greetings: don't call OpenAI or touch vectors at all.
    normalized_q = question.strip().lower()
    # Remove common punctuation at the end like ?, !, .
    while normalized_q and normalized_q[-1] in {'?', '!', '.', ','}:
        normalized_q = normalized_q[:-1].strip()

    if normalized_q in {'hi', 'hello', 'hey', 'hellow'}:
        return {
            'answer': 'Hi! Ask me anything about this video and I will answer based on its content.',
            'sources': [],
            'cached': False,
            'session_id': None,
            'message_id': None,
        }
    # Single embedding for both cache lookup and vector search (saves ~0.5–2s)
    start_time = time.time()
    question_embedding = generate_embedding(question)
    embedding_time = time.time() - start_time
    
    cache_start = time.time()
    cached_response = check_cache(video_id, question, question_embedding=question_embedding)
    cache_time = time.time() - cache_start
    if cached_response:
        # Fast path: return cached answer immediately (skip DB writes for speed)
        return {
            'answer': cached_response['answer'],
            'sources': cached_response['sources'],
            'cached': True,
            'session_id': session_id,
            'message_id': None,
        }

    # Retrieve top 2 chunks for better context coverage (roles info may span chunks)
    search_start = time.time()
    relevant_chunks = vector_search(question_embedding, video_id, top_k=2)
    search_time = time.time() - search_start
    
    if not relevant_chunks:
        raise Exception('No relevant content found for this question')
    
    context = build_context(relevant_chunks)
    llm_start = time.time()
    answer = get_llm_response(question, context)
    llm_time = time.time() - llm_start
    
    total_time = time.time() - start_time
    print(f'[PERF] Embedding: {embedding_time:.2f}s, Cache: {cache_time:.2f}s, Search: {search_time:.2f}s, LLM: {llm_time:.2f}s, Total: {total_time:.2f}s')
    
    # Format sources
    sources = [
        {
            'text': chunk.text,
            'timestamp': format_timestamp(chunk.start_time),
            'start_time': chunk.start_time,
            'end_time': chunk.end_time
        }
        for chunk in relevant_chunks
    ]

    # Prepare response first so user is not blocked by cache writes
    response = {
        'answer': answer,
        'sources': sources,
        'cached': False,
        'session_id': session_id,
        'message_id': None,
    }

    # Save to QuestionCache in background so next similar question is instant
    def _save_to_cache():
        try:
            expires_at = datetime.now() + timedelta(days=30)
            cache_entry = QuestionCache.objects.create(
                video_id=video_id,
                question=question,
                question_embedding=json.dumps(question_embedding),
                answer=answer,
                expires_at=expires_at,
            )
            cache_entry.source_chunks.set(relevant_chunks)
        except Exception as e:
            # Don't crash the request if caching fails
            print(f'[v0] Cache save error: {str(e)}')

    # Fire-and-forget cache save
    threading.Thread(target=_save_to_cache, daemon=True).start()

    return response


def ask_ai_service_stream(user, video_id: str, question: str, session_id: int = None):
    """
    Generator for streaming Ask AI. Yields dicts for SSE:
      {"type": "meta", "sources": [...], "cached": bool}
      {"type": "content", "content": "chunk"}
      {"type": "done", "answer": "...", "sources": [...]}
    """
    normalized_q = question.strip().lower()
    while normalized_q and normalized_q[-1] in {'?', '!', '.', ','}:
        normalized_q = normalized_q[:-1].strip()

    if normalized_q in {'hi', 'hello', 'hey', 'hellow'}:
        yield {'type': 'meta', 'sources': [], 'cached': False}
        yield {'type': 'content', 'content': 'Hi! Ask me anything about this video and I will answer based on its content.'}
        yield {'type': 'done', 'answer': 'Hi! Ask me anything about this video and I will answer based on its content.', 'sources': []}
        return

    question_embedding = generate_embedding(question)
    cached_response = check_cache(video_id, question, question_embedding=question_embedding)

    if cached_response:
        yield {'type': 'meta', 'sources': cached_response['sources'], 'cached': True}
        yield {'type': 'content', 'content': cached_response['answer']}
        yield {'type': 'done', 'answer': cached_response['answer'], 'sources': cached_response['sources']}
        return

    relevant_chunks = vector_search(question_embedding, video_id, top_k=2)
    if not relevant_chunks:
        yield {'type': 'error', 'message': 'No relevant content found for this question'}
        return

    context = build_context(relevant_chunks)
    sources = [
        {'text': chunk.text, 'timestamp': format_timestamp(chunk.start_time), 'start_time': chunk.start_time, 'end_time': chunk.end_time}
        for chunk in relevant_chunks
    ]
    yield {'type': 'meta', 'sources': sources, 'cached': False}

    full_answer_parts = []
    for chunk_text in get_llm_response_stream(question, context):
        full_answer_parts.append(chunk_text)
        yield {'type': 'content', 'content': chunk_text}

    full_answer = ''.join(full_answer_parts).strip()
    yield {'type': 'done', 'answer': full_answer, 'sources': sources}

    def _save_to_cache():
        try:
            expires_at = datetime.now() + timedelta(days=30)
            cache_entry = QuestionCache.objects.create(
                video_id=video_id,
                question=question,
                question_embedding=json.dumps(question_embedding),
                answer=full_answer,
                expires_at=expires_at,
            )
            cache_entry.source_chunks.set(relevant_chunks)
        except Exception as e:
            print(f'[v0] Cache save error: {str(e)}')
    threading.Thread(target=_save_to_cache, daemon=True).start()


def ingest_transcript(video_id: str, transcript: str, video_title: str | None = None) -> None:
    """
    Ingest a full transcript for a given video_id coming from Bubble.
    This splits the transcript into chunks, generates embeddings, and stores them.
    """
    # Remove any existing chunks for this video to avoid duplicates
    TranscriptChunk.objects.filter(video_id=video_id).delete()

    chunks = split_transcript(transcript, chunk_size=300, chunk_overlap=50)

    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk['text'])
        TranscriptChunk.objects.create(
            video_id=video_id,
            video_title=video_title or '',
            text=chunk['text'],
            start_time=chunk['start_time'],
            end_time=chunk['end_time'],
            embedding=json.dumps(embedding),
            sequence_number=i,
        )


# ========== Pinecone Vector Search Functions ==========

def ingest_transcript_to_pinecone(video_id: str, transcript: str, video_title: str | None = None) -> dict:
    """
    Ingest a full transcript directly to Pinecone (not SQLite).
    Splits transcript, generates embeddings, and upserts to Pinecone with video_id/title in metadata.
    
    Returns:
        dict with status and chunk count
    """
    get_pinecone_index()  # Will raise exception if not configured
    
    # Split transcript into chunks
    chunks = split_transcript(transcript, chunk_size=300, chunk_overlap=50)
    
    # Generate embeddings for all chunks
    embeddings = []
    for chunk in chunks:
        embedding = generate_embedding(chunk['text'])
        embeddings.append(embedding)
    
    # Upsert to Pinecone
    pinecone_upsert_chunks(video_id, video_title or '', chunks, embeddings)
    
    return {
        'status': 'ok',
        'chunks_ingested': len(chunks),
        'video_id': video_id,
        'video_title': video_title or '',
    }


def pinecone_upsert_chunks(video_id: str, video_title: str, chunks: list, embeddings: list):
    """
    Upsert transcript chunks to Pinecone with video_id and title in metadata.
    
    Args:
        video_id: Video identifier
        video_title: Video title
        chunks: List of chunk dicts with 'text', 'start_time', 'end_time', 'sequence_number'
        embeddings: List of embedding vectors (same length as chunks)
    """
    index = get_pinecone_index()
    
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # split_transcript does not include sequence_number; use loop index.
        sequence_number = i
        vector_id = f"{video_id}_{sequence_number}"
        vectors.append({
            'id': vector_id,
            'values': embedding,
            'metadata': {
                'video_id': video_id,
                'video_title': video_title or '',
                'text': chunk['text'],
                'start_time': chunk['start_time'],
                'end_time': chunk['end_time'],
                'sequence_number': sequence_number,
            }
        })
    
    # Upsert in batches (Pinecone limit is usually 100 vectors per request)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)


def pinecone_vector_search(question_embedding: list, video_id: str, top_k: int = 3):
    """
    Search Pinecone for relevant chunks filtered by video_id.
    
    Returns:
        List of dicts with 'text', 'start_time', 'end_time', 'score', 'metadata'
    """
    index = get_pinecone_index()  # Will raise exception if not configured
    
    # Query Pinecone with metadata filter for video_id
    results = index.query(
        vector=question_embedding,
        top_k=top_k,
        include_metadata=True,
        filter={'video_id': {'$eq': video_id}}
    )
    
    chunks = []
    for match in results.matches:
        metadata = match.metadata or {}
        chunks.append({
            'text': metadata.get('text', ''),
            'start_time': metadata.get('start_time', 0),
            'end_time': metadata.get('end_time', 0),
            'score': match.score,
            'metadata': metadata,
        })
    
    return chunks


def ask_ai_service_pinecone(user, video_id: str, question: str, session_id: int = None) -> dict:
    """
    Ask AI service using Pinecone for vector search with caching for repeated questions.
    Same flow as ask_ai_service but uses Pinecone instead of SQLite.
    """
    import time
    start_time = time.time()
    
    # Fast-path for greetings
    normalized_q = question.strip().lower()
    while normalized_q and normalized_q[-1] in {'?', '!', '.', ','}:
        normalized_q = normalized_q[:-1].strip()
    if normalized_q in {'hi', 'hello', 'hey', 'hellow'}:
        return {
            'answer': 'Hi! Ask me anything about this video and I will answer based on its content.',
            'sources': [],
            'cached': False,
            'session_id': session_id,
            'message_id': None,
            'response_time_ms': int((time.time() - start_time) * 1000),
        }
    
    # Generate embedding (needed for both cache check and Pinecone search)
    embedding_start = time.time()
    question_embedding = generate_embedding(question)
    embedding_time = time.time() - embedding_start
    
    # Check cache first (reuse embedding to avoid duplicate API call)
    cache_start = time.time()
    cached_response = check_cache(video_id, question, question_embedding=question_embedding)
    cache_time = time.time() - cache_start
    
    if cached_response:
        # Fast path: return cached answer immediately
        total_time = time.time() - start_time
        return {
            'answer': cached_response['answer'],
            'sources': cached_response['sources'],
            'cached': True,
            'session_id': session_id,
            'message_id': None,
            'response_time_ms': int(total_time * 1000),
            'performance': {
                'embedding_ms': int(embedding_time * 1000),
                'cache_ms': int(cache_time * 1000),
                'search_ms': 0,
                'llm_ms': 0,
                'total_ms': int(total_time * 1000),
            }
        }
    
    # Search Pinecone
    search_start = time.time()
    relevant_chunks_data = pinecone_vector_search(question_embedding, video_id, top_k=2)
    search_time = time.time() - search_start
    
    if not relevant_chunks_data:
        raise Exception('No relevant content found for this question in Pinecone')
    
    # Build context from Pinecone results
    context_parts = []
    for chunk_data in relevant_chunks_data:
        text = chunk_data['text'].strip()
        import re
        text = re.sub(r'\d{2}:\d{2}\s*', '', text)
        text = re.sub(r'\s+', ' ', text)
        context_parts.append(text.strip())
    context = '\n\n---\n\n'.join(context_parts)
    
    # Get LLM response
    llm_start = time.time()
    answer = get_llm_response(question, context)
    llm_time = time.time() - llm_start
    
    total_time = time.time() - start_time
    print(f'[PERF-Pinecone] Embedding: {embedding_time:.2f}s, Cache: {cache_time:.2f}s, Search: {search_time:.2f}s, LLM: {llm_time:.2f}s, Total: {total_time:.2f}s')
    
    # Format sources
    sources = [
        {
            'text': chunk_data['text'],
            'timestamp': format_timestamp(chunk_data['start_time']),
            'start_time': chunk_data['start_time'],
            'end_time': chunk_data['end_time'],
            'score': chunk_data.get('score', 0),
        }
        for chunk_data in relevant_chunks_data
    ]
    
    response = {
        'answer': answer,
        'sources': sources,
        'cached': False,
        'session_id': session_id,
        'message_id': None,
        'response_time_ms': int(total_time * 1000),
        'performance': {
            'embedding_ms': int(embedding_time * 1000),
            'cache_ms': int(cache_time * 1000),
            'search_ms': int(search_time * 1000),
            'llm_ms': int(llm_time * 1000),
            'total_ms': int(total_time * 1000),
        }
    }
    
    # Save to cache in background (reuse question_embedding to avoid re-computing)
    def _save_to_cache():
        try:
            # Convert Pinecone chunks to TranscriptChunk-like objects for cache storage
            # We need to create TranscriptChunk objects or store minimal data
            # For now, we'll store the answer and question embedding; sources can be reconstructed
            expires_at = datetime.now() + timedelta(days=30)
            cache_entry = QuestionCache.objects.create(
                video_id=video_id,
                question=question,
                question_embedding=json.dumps(question_embedding),
                answer=answer,
                expires_at=expires_at,
            )
            # Note: QuestionCache.source_chunks expects TranscriptChunk objects
            # Since we're using Pinecone, we can't link to TranscriptChunk directly
            # The cache will still work for answer lookup, but source_chunks will be empty
            # This is fine - the answer is what matters for cache hits
        except Exception as e:
            print(f'[Pinecone] Cache save error: {str(e)}')
    
    threading.Thread(target=_save_to_cache, daemon=True).start()
    
    return response
