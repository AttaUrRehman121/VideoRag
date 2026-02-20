import json
from django.http import StreamingHttpResponse
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from drf_spectacular.utils import extend_schema
from .models import TranscriptChunk, QuestionCache
from .serializers import (
    TranscriptChunkSerializer,
    AskAIRequestSerializer,
    AskAIResponseSerializer,
    TranscriptIngestSerializer,
)


class TranscriptChunkViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only endpoint for transcript chunks, plus an ingest endpoint for Bubble."""
    queryset = TranscriptChunk.objects.all()
    serializer_class = TranscriptChunkSerializer
    # Bubble calls this directly; no auth required.
    permission_classes = [AllowAny]

    def get_queryset(self):
        """Filter chunks by external video_id if provided."""
        queryset = TranscriptChunk.objects.all()
        video_id = self.request.query_params.get('video_id')
        if video_id:
            queryset = queryset.filter(video_id=video_id)
        return queryset

    @extend_schema(
        request=TranscriptIngestSerializer,
        responses={'200': {'type': 'object'}},
    )
    @action(detail=False, methods=['post'], url_path='ingest')
    def ingest(self, request):
        """
        Ingest transcript for a video coming from Bubble.

        Expected body:
        {
          "video_id": "bubble-video-id",
          "transcript": "full transcript text",
          "video_title": "Optional title"
        }
        """
        serializer = TranscriptIngestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        from .services import ingest_transcript

        ingest_transcript(
            video_id=serializer.validated_data['video_id'],
            transcript=serializer.validated_data['transcript'],
            video_title=serializer.validated_data.get('video_title'),
        )

        return Response({'status': 'ok'}, status=status.HTTP_200_OK)

    @extend_schema(
        request=TranscriptIngestSerializer,
        responses={'200': {'type': 'object'}},
    )
    @action(detail=False, methods=['post'], url_path='ingest_pinecone')
    def ingest_pinecone(self, request):
        """
        Ingest transcript directly to Pinecone (not SQLite).
        Splits transcript, generates embeddings, and upserts to Pinecone with video_id/title in metadata.

        Expected body:
        {
          "video_id": "bubble-video-id",
          "transcript": "full transcript text",
          "video_title": "Optional title"
        }
        """
        serializer = TranscriptIngestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        from .services import ingest_transcript_to_pinecone

        try:
            result = ingest_transcript_to_pinecone(
                video_id=serializer.validated_data['video_id'],
                transcript=serializer.validated_data['transcript'],
                video_title=serializer.validated_data.get('video_title'),
            )
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AskAIViewSet(viewsets.ViewSet):
    """Handles Ask AI requests with RAG capabilities."""
    # Bubble calls this directly; no auth required.
    permission_classes = [AllowAny]

    @extend_schema(
        request=AskAIRequestSerializer,
        responses=AskAIResponseSerializer,
    )
    @action(detail=False, methods=['post'])
    def ask(self, request):
        """
        Main Ask AI endpoint.
        
        Request body:
        {
            "video_id": 123,
            "question": "What is the main topic of the video?",
            "session_id": 456 (optional)
        }
        """
        serializer = AskAIRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        from .services import ask_ai_service

        try:
            response = ask_ai_service(
                user=request.user,
                video_id=serializer.validated_data['video_id'],
                question=serializer.validated_data['question'],
                session_id=serializer.validated_data.get('session_id')
            )
            response_serializer = AskAIResponseSerializer(response)
            return Response(response_serializer.data)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

    @extend_schema(
        request=AskAIRequestSerializer,
        responses={200: {'description': 'SSE stream of events (meta, content, done)'}},
    )
    @action(detail=False, methods=['post'], url_path='ask_stream')
    def ask_stream(self, request):
        """
        Ask AI with streaming response (Server-Sent Events).
        POST same body as /ask/; response is text/event-stream.
        Events: data: {"type":"meta",...} then data: {"type":"content","content":"..."} then data: {"type":"done",...}
        """
        serializer = AskAIRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        def event_stream():
            from .services import ask_ai_service_stream
            try:
                for event in ask_ai_service_stream(
                    user=request.user,
                    video_id=serializer.validated_data['video_id'],
                    question=serializer.validated_data['question'],
                    session_id=serializer.validated_data.get('session_id'),
                ):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        response = StreamingHttpResponse(
            event_stream(),
            content_type='text/event-stream',
        )
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['X-Accel-Buffering'] = 'no'
        return response

    @extend_schema(
        request=AskAIRequestSerializer,
        responses={200: {'description': 'Pinecone-based Ask AI response with performance metrics'}},
    )
    @action(detail=False, methods=['post'], url_path='ask_pinecone')
    def ask_pinecone(self, request):
        """
        Ask AI endpoint using Pinecone for vector search (for testing/performance comparison).
        Same request body as /ask/ but uses Pinecone instead of SQLite for vector search.
        Returns response with performance metrics (embedding_ms, search_ms, llm_ms, total_ms).
        
        Request body:
        {
            "video_id": "123",
            "question": "What is the main topic of the video?",
            "session_id": 456 (optional)
        }
        """
        serializer = AskAIRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        from .services import ask_ai_service_pinecone

        try:
            response = ask_ai_service_pinecone(
                user=request.user,
                video_id=serializer.validated_data['video_id'],
                question=serializer.validated_data['question'],
                session_id=serializer.validated_data.get('session_id')
            )
            return Response(response)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
