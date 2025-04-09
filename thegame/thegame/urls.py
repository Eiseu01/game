from django.contrib import admin
from django.urls import path

from base.views import home, result

urlpatterns = [
    path('', home, name='home'),
    path('result/', result, name='result')
]
