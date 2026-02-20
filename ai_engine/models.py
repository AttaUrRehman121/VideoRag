from django.db import models


class TranscriptChunk(models.Model):
    """
    Chunks of transcripts with vector embeddings for similarity search.
    Embeddings stored as JSON in TextField (SQLite-compatible).
    """
    video_id = models.CharField(max_length=255, db_index=True)
    video_title = models.CharField(max_length=255, blank=True, null=True)
    text = models.TextField()
    start_time = models.IntegerField(help_text='Start time in seconds')
    end_time = models.IntegerField(help_text='End time in seconds')
    embedding = models.TextField(help_text='JSON list of 1536 floats from text-embedding-3-small')
    sequence_number = models.IntegerField(help_text='Order in the transcript')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Transcript Chunk'
        verbose_name_plural = 'Transcript Chunks'
        ordering = ['video_id', 'sequence_number']
        indexes = [
            models.Index(fields=['video_id', 'sequence_number']),
        ]

    def __str__(self):
        title = self.video_title or self.video_id
        return f'{title} - Chunk {self.sequence_number}'


class QuestionCache(models.Model):
    """Cache for frequently asked questions and answers."""
    video_id = models.CharField(max_length=255, db_index=True)
    question = models.TextField()
    question_embedding = models.TextField(help_text='JSON list of 1536 floats')
    answer = models.TextField()
    source_chunks = models.ManyToManyField(TranscriptChunk, related_name='cached_in')
    accessed_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    expires_at = models.DateTimeField(help_text='Cache expiration time')

    class Meta:
        verbose_name = 'Question Cache'
        verbose_name_plural = 'Question Cache'
        ordering = ['-accessed_count', '-created_at']
        indexes = [
            models.Index(fields=['video_id', 'created_at']),
        ]

    def __str__(self):
        return f'Cache: {self.question[:50]}...'
