from django.contrib import admin
from django.urls import path
from Accounts import views

urlpatterns = [
    path('slogin',views.slogin,name="slogin"),
    path('tlogin',views.tlogin,name="tlogin"),
    path('logout',views.logout,name="logout"),
    path('import_users', views.import_users, name='import_users'),
]

