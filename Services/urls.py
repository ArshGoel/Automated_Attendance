from django.contrib import admin
from django.urls import path
from Services import views

urlpatterns = [
    path('student',views.student,name="student"),
    path('teacher',views.teacher,name="teacher"),
    # path('take_attendance/', views.take_attendance, name='take_attendance'),
    # path('logout',views.logout,name="logout"),
    path('attendance/', views.attendance, name='attendance'),
    path('save_attendance', views.save_attendance, name='save_attendance'),
    path('video-capture/', views.video_capture, name='video_capture'),

]




