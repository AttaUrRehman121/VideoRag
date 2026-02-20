"""
URL configuration for Video RAG AI system.
"""
from django.contrib import admin
from django.http import HttpResponse
from django.urls import path, include
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView


def empty_firebase_sw(_request):
    """
    Dummy firebase-messaging-sw.js so browsers stop logging 404 warnings.
    You are not using Firebase messaging, so this is an empty no-op script.
    """
    return HttpResponse('// no firebase messaging used here\n', content_type='application/javascript')


urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Core AI URLs
    path('api/ai/', include('ai_engine.urls')),

    # Dummy Firebase service worker to avoid 404 warnings
    path('firebase-messaging-sw.js', empty_firebase_sw),

    # API schema & documentation
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/docs/swagger/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/docs/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
]
