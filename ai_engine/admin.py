from django.contrib import admin
from .models import TranscriptChunk, QuestionCache


@admin.register(TranscriptChunk)
class TranscriptChunkAdmin(admin.ModelAdmin):
    list_display = ('video_id', 'video_title', 'sequence_number', 'start_time', 'end_time', 'created_at')
    search_fields = ('video_id', 'video_title', 'text')
    list_filter = ('video_id', 'created_at')
    ordering = ('video_id', 'sequence_number')


@admin.register(QuestionCache)
class QuestionCacheAdmin(admin.ModelAdmin):
    list_display = ('video_id', 'question', 'accessed_count', 'created_at', 'expires_at')
    search_fields = ('question', 'video_id')
    list_filter = ('video_id', 'created_at', 'accessed_count')
    readonly_fields = ('created_at', 'updated_at')
    ordering = ('-accessed_count', '-created_at')
