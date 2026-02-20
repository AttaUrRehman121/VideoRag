from rest_framework import serializers
from .models import TranscriptChunk, QuestionCache


class TranscriptChunkSerializer(serializers.ModelSerializer):
    start_time_formatted = serializers.SerializerMethodField()
    end_time_formatted = serializers.SerializerMethodField()

    class Meta:
        model = TranscriptChunk
        fields = [
            'id',
            'video_id',
            'video_title',
            'text',
            'start_time',
            'start_time_formatted',
            'end_time',
            'end_time_formatted',
            'sequence_number',
            'created_at',
        ]
        read_only_fields = ['id', 'created_at']

    def get_start_time_formatted(self, obj):
        return self._format_time(obj.start_time)

    def get_end_time_formatted(self, obj):
        return self._format_time(obj.end_time)

    @staticmethod
    def _format_time(seconds):
        """Convert seconds to MM:SS format."""
        minutes = seconds // 60
        secs = seconds % 60
        return f'{minutes:02d}:{secs:02d}'


class QuestionCacheSerializer(serializers.ModelSerializer):
    class Meta:
        model = QuestionCache
        fields = [
            'id',
            'video_id',
            'question',
            'answer',
            'accessed_count',
            'created_at',
            'updated_at',
            'expires_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'accessed_count']


class AskAIRequestSerializer(serializers.Serializer):
    """Serializer for Ask AI API endpoint."""
    video_id = serializers.CharField(max_length=255)
    question = serializers.CharField(max_length=2000)
    session_id = serializers.IntegerField(required=False, allow_null=True)


class AskAIResponseSerializer(serializers.Serializer):
    """Serializer for Ask AI API response."""
    answer = serializers.CharField()
    sources = serializers.ListField(child=serializers.DictField())
    cached = serializers.BooleanField()
    session_id = serializers.IntegerField(required=False, allow_null=True)
    message_id = serializers.IntegerField(required=False, allow_null=True)


class TranscriptIngestSerializer(serializers.Serializer):
    """
    Serializer for ingesting a transcript for a given video from Bubble.
    Bubble sends us the video_id and full transcript text.
    """
    video_id = serializers.CharField(max_length=255)
    transcript = serializers.CharField()
    video_title = serializers.CharField(max_length=255, required=False, allow_blank=True)
