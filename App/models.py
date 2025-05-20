from django.db import models
from django.contrib.auth import get_user_model
import datetime

User = get_user_model()   

class Profile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    
    def __str__(self):
        return str(self.user)
    
class Subtitle(models.Model):
    user_profile = models.ForeignKey(Profile, on_delete=models.CASCADE)
    audio = models.FileField(upload_to='Audio', max_length=100, null=True)
    subtitle = models.FileField(upload_to='Subtitle', max_length=100, null=True)
    webVit = models.FileField(upload_to='Subtitle', max_length=100, null=True)
    source_lang = models.CharField(max_length=100, null=True)
    target_lang = models.CharField(max_length=100, null=True)
    duration = models.CharField(max_length=100, null=True)
    created_at = models.DateTimeField(auto_now_add=True)   
    updated_at = models.DateTimeField(auto_now=True)  
    processing_duration = models.CharField(max_length=50, default=0)
    status = models.CharField(max_length=100, null=True, default='Pending',choices=[
            ('pending', 'Pending'),
            ('processing', 'Processing'),
            ('completed', 'Completed'),
            ('failed', 'Failed')])
    
    is_deleted = models.BooleanField(default=False)
    is_completed = models.BooleanField(default=False)
    actions = models.CharField(max_length=100, null=True, default='Pending')
    
    
    def __str__(self):
        return str(self.audio)