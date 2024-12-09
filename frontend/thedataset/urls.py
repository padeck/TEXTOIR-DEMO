"""the dataset  URL Configuration

"""
from django.urls import re_path
from django.contrib import admin
from django.urls import path

from . import views
urlpatterns = [
    path('toDatasetList/',views.toDatasetList),
    path('getDatasetList/',views.getDatasetList),
    path('toAddHtml/',views.toAddHtml),
    re_path(r'^addDataset/$',views.addDataset),
    re_path(r'^details/$',views.details),
    re_path(r'^downloadDataset/$',views.downloadDataset),
    re_path(r'^toEdit/$', views.toEdit),
    re_path(r'^editData/$', views.editData),
    re_path(r'^delData/$', views.delData),
    re_path(r'^update_source/$', views.update_source),

    
]
