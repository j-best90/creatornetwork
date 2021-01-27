from django.db import models
import uuid

# Create your models here.
class creatorDetails(models.Model):
    creator_name = models.CharField(max_length=75,default="")
    creator_id = models.UUIDField(primary_key=True, default=uuid.uuid4,editable=False)
    primary_content_category = models.CharField(max_length=100,default=None)
    secondary_content_category = models.CharField(max_length=100,default=None)
    posts_per_week_ig = models.IntegerField(default=0,blank=True,null=True)
    rate_card = models.DecimalField(max_digits=10, decimal_places=2,default=None)
    ig_audience = models.IntegerField(default=None,blank=True,null=True)
    fb_audience = models.IntegerField(default=None,blank=True,null=True)
    tw_audience = models.IntegerField(default=None,blank=True,null=True)
    yt_audience = models.IntegerField(default=None,blank=True,null=True)

    def __str__(self):
        return '{}'.format(self.creator_name)

class creatorProfiles(models.Model):
    creator_name = models.CharField(max_length=75,default="")
    creator_id = models.CharField(max_length=75,default="")
    gender = models.CharField(max_length=7,default=None)
    email_address = models.EmailField(max_length=254,default=None,blank=True,null=True)
    ig_profile_link = models.URLField(max_length=300,default=None,blank=True,null=True)
    fb_profle_link = models.URLField(max_length=300,default=None,blank=True,null=True)
    yt_profile_link = models.URLField(max_length=300,default=None,blank=True,null=True)
    tw_profile_link = models.URLField(max_length=300,default=None,blank=True,null=True)

    def __str__(self):
        return '{}'.format(self.creator_name)

class creatorPerformance(models.Model):
    creator_name = models.CharField(max_length=75,default="")
    creator_id = models.CharField(max_length=75,default="")
    share_YN = models.IntegerField(default=None,blank=True,null=True)
    share_volume = models.IntegerField(default=None,blank=True,null=True)
    IG_share = models.IntegerField(default=None,blank=True,null=True)
    FB_share = models.IntegerField(default=None,blank=True,null=True)
    YT_share = models.IntegerField(default=None,blank=True,null=True)
    TW_share = models.IntegerField(default=None,blank=True,null=True)

    def __str__(self):
        return '{}'.format(self.creator_name)

class creatorSocialStats(models.Model):
    creator_name = models.CharField(max_length=75,default="")
    creator_id = models.CharField(max_length=75,default="")
    reach_per_post = models.IntegerField(default=None,blank=True,null=True)
    perc_audience_reached = models.FloatField(default=None,blank=True,null=True)
    reach_per_story = models.IntegerField(default=None,blank=True,null=True)
    per_audience_reached_story = models.FloatField(default=None,blank=True,null=True)

    def __str__(self):
        return '{}'.format(self.creator_name)

class youtubeStats(models.Model):
    creator_name = models.CharField(max_length=75,default="")
    creator_id = models.CharField(max_length=75,default="")
    outbound_post = models.CharField(max_length=350,default=None)
    status = models.CharField(max_length=20,default=None)
    account = models.CharField(max_length=75,default=None)
    account_type = models.CharField(max_length=25,default=None)
    author_name = models.CharField(max_length=50,default=None,blank=True,null=True)
    campaign_name = models.CharField(max_length=75,default=None)
    permalink = models.URLField(max_length=300,default=None,blank=True,null=True)
    scheduled_time = models.DateTimeField(auto_now=False,auto_now_add=False,default=None)
    published_time = models.DateTimeField(auto_now=False,auto_now_add=False,default=None)
    tags = models.CharField(max_length=50,default=None,blank=True,null=True)
    youtube_video_views_sum = models.IntegerField(default=None,blank=True,null=True)
    youtube_video_comments_sum = models.IntegerField(default=None,blank=True,null=True)
    youtube_video_likes_sum = models.IntegerField(default=None,blank=True,null=True)
    youtube_video_dislikes_sum = models.IntegerField(default=None,blank=True,null=True)
    youtube_video_average_view_duration_sum = models.TimeField(default=None,blank=True,null=True)
    youtube_video_average_view_percentage_in_perc = models.TimeField(default=None,blank=True,null=True)

    def __str__(self):
        return '{}'.format(self.creator_name)

class creatorBio(models.Model):
    creator_name = models.CharField(max_length=75, default="", editable=False)
    creator_id = models.CharField(max_length=75,default="")

    def __str__(self):
        return '{}'.format(self.creator_name)