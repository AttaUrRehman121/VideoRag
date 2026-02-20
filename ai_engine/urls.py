from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import TranscriptChunkViewSet, AskAIViewSet

router = DefaultRouter()
router.register(r'chunks', TranscriptChunkViewSet, basename='transcript-chunk')
router.register(r'ask', AskAIViewSet, basename='ask-ai')

urlpatterns = [
    path('', include(router.urls)),
]
