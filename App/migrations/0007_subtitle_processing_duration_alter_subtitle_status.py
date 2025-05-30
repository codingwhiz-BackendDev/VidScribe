# Generated by Django 5.0 on 2025-04-22 03:05

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('App', '0006_subtitle_created_at_subtitle_updated_at'),
    ]

    operations = [
        migrations.AddField(
            model_name='subtitle',
            name='processing_duration',
            field=models.CharField(default=django.utils.timezone.now, max_length=50),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='subtitle',
            name='status',
            field=models.CharField(choices=[('pending', 'Pending'), ('processing', 'Processing'), ('completed', 'Completed'), ('failed', 'Failed')], default='Pending', max_length=100, null=True),
        ),
    ]
