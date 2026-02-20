"""
Management command to initialize pgvector extension in PostgreSQL.
Run with: python manage.py init_pgvector
"""
from django.core.management.base import BaseCommand
from django.db import connection


class Command(BaseCommand):
    help = 'Initialize pgvector extension in PostgreSQL database'

    def handle(self, *args, **options):
        with connection.cursor() as cursor:
            try:
                # Create pgvector extension
                cursor.execute('CREATE EXTENSION IF NOT EXISTS vector;')
                self.stdout.write(
                    self.style.SUCCESS('Successfully initialized pgvector extension')
                )
                
                # Create IVFFLAT index on embedding columns
                # This will be done after migrations
                self.stdout.write(
                    self.style.SUCCESS('pgvector is ready for use')
                )
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error initializing pgvector: {str(e)}')
                )
