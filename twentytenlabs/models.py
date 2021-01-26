from django.db import models
import uuid

# Create your models here.
class creatorDetails(models.Model):
    creator_id = models.UUIDField(primary_key=True, default=uuid.uuid4,editable=False)
    creator_name = models.CharField(max_length=75)
    primary_content_category = models.CharField(max_length=100)
    secondary_content_category = models.CharField(max_length=100)
    posts_per_week_ig = models.IntegerField()
    rate_card = models.DecimalField(max_digits=10, decimal_places=2)
    ig_audience = models.IntegerField()
    fb_audience = models.IntegerField()
    tw_audience = models.IntegerField()
    yt_audience = models.IntegerField()

    def __str__(self):
        return '{}'.format(self.creator_name)

class creatorProfiles(models.Model):
    pass

class creatorPerformance(models.Model):
    pass

class creatorSocialStats(models.Model):
    pass

class youtubeStats(models.Model):
    pass

class creatorBio(models.Model):
    pass