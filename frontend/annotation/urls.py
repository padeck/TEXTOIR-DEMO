from django.urls import re_path
from django.urls import path
from . import views
urlpatterns = [
    re_path(r'^data_annotation/$',views.data_annotation),
    re_path(r'^getExampleList/$',views.getExampleList),
    re_path(r'^updateResultByResultId/$',views.updateResultByResultId),
    re_path(r'^judgeStateDataAnnotation/$',views.judgeStateDataAnnotation),
    re_path(r'^judgeStateDataExport2Disk/$',views.judgeStateDataExport2Disk),
    re_path(r'^getDatasetList/$',views.getDatasetList),
    re_path(r'^getClassListByDatasetNameAndClassType/$',views.getClassListByDatasetNameAndClassType),
    re_path(r'^getTextListByDatasetClassTypeLabelName/$',views.getTextListByDatasetClassTypeLabelName),
    re_path(r'^getTextListByDatasetForUnknown/$',views.getTextListByDatasetForUnknown),
]