# SQLite-compatible initial migration (embeddings stored as JSON TextField)

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='TranscriptChunk',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('video_id', models.CharField(db_index=True, max_length=255)),
                ('video_title', models.CharField(blank=True, max_length=255, null=True)),
                ('text', models.TextField()),
                ('start_time', models.IntegerField(help_text='Start time in seconds')),
                ('end_time', models.IntegerField(help_text='End time in seconds')),
                ('embedding', models.TextField(help_text='JSON list of 1536 floats from text-embedding-3-small')),
                ('sequence_number', models.IntegerField(help_text='Order in the transcript')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'verbose_name': 'Transcript Chunk',
                'verbose_name_plural': 'Transcript Chunks',
                'ordering': ['video_id', 'sequence_number'],
            },
        ),
        migrations.CreateModel(
            name='QuestionCache',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('video_id', models.CharField(db_index=True, max_length=255)),
                ('question', models.TextField()),
                ('question_embedding', models.TextField(help_text='JSON list of 1536 floats')),
                ('answer', models.TextField()),
                ('accessed_count', models.IntegerField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('expires_at', models.DateTimeField(help_text='Cache expiration time')),
                ('source_chunks', models.ManyToManyField(related_name='cached_in', to='ai_engine.transcriptchunk')),
            ],
            options={
                'verbose_name': 'Question Cache',
                'verbose_name_plural': 'Question Cache',
                'ordering': ['-accessed_count', '-created_at'],
            },
        ),
        migrations.AddIndex(
            model_name='transcriptchunk',
            index=models.Index(fields=['video_id', 'sequence_number'], name='ai_engine_t_video_i_0c11a8_idx'),
        ),
        migrations.AddIndex(
            model_name='questioncache',
            index=models.Index(fields=['video_id', 'created_at'], name='ai_engine_q_video_i_5232a1_idx'),
        ),
    ]
