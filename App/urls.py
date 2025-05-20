from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload', views.upload, name='upload'),
    path('login', views.login, name='login'),
    path('register', views.register, name='register'),
    path('view_subtitle/<int:pk>', views.view_subtitle, name='view_subtitle'),
    path('subtitle/<int:pk>/', views.view_subtitle, name='view_subtitle'),
    path('download_subtitle_format/<int:pk>/<str:format_type>/', views.download_subtitle_format, name='download_subtitle_format'),
    path('save-transcript/<int:pk>/', views.save_transcript, name='save_transcript'),
    path('dashboard', views.dashboard,  name='dashboard'),
    path('summarize', views.summarize,  name='summarize'),

]
