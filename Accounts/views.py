from django.shortcuts import render,redirect
from django.contrib.auth import authenticate,get_user_model
from django.contrib.auth.models  import User
from django.contrib import auth,messages
from django.contrib.auth import authenticate, login as auth_login
from django.shortcuts import render, redirect
import csv
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
import os

def slogin(request):
    if request.method == "POST":
        form_type = request.POST.get("form_type")  # Check which form was submitted

        if form_type == "login":
            # Handle login form submission
            username = request.POST.get("username")
            password = request.POST.get("password")
            if username and password:
                user = authenticate(username=username, password=password)
                if user is not None:
                    if user.is_superuser:
                        messages.warning(request, "Please Login Using Teacher Login!!")
                        return redirect("/auth/tlogin")
                    else:
                        auth_login(request, user)
                        messages.success(request, "Login successful")
                        return redirect('/dashboard/student')
                else:
                    messages.error(request, "Invalid credentials")
            return render(request, "student.html", {"username": username})

        elif form_type == "signup":
            username = request.POST.get("username")
            password = request.POST.get("password")
            email = request.POST.get("email")
            try:
                user = User.objects.create_user(username = username , password = password)
                user.save()
                auth.login(request ,user)

            except:
                messages.error(request,"Username Already exists")
                return render(request,"student.html",{"username":username,"email":email}) 
                
            messages.success(request,"Success")
    return render(request, "student.html")


def tlogin(request):
    if request.method == "POST":
        form_type = request.POST.get("form_type")  # Check which form was submitted

        if form_type == "login":
            username = request.POST.get("username")
            password = request.POST.get("password")
            if username and password:
                user = authenticate(username=username, password=password)
                if user is not None:
                    if user.is_superuser:
                        # Handle superuser login
                        auth_login(request, user)
                        messages.success(request, "Superuser login successful")
                        return redirect('/dashboard/teacher')  # Redirect to superuser dashboard or another page
                    else:
                        messages.warning(request, "Please Login Using Student Login!!")
                        return redirect('/auth/slogin')
                else:
                    messages.error(request, "Invalid credentials")
                    return render(request, "teacher.html", {"username": username})

        elif form_type == "signup":
            # Handle signup form submission
            username = request.POST.get("username")
            password = request.POST.get("password")
            email = request.POST.get("email")
            try:
                user = User.objects.create_superuser(username = username , password = password,email = email)
                user.save()
                auth.login(request ,user)

            except:
                messages.error(request,"Username Already exists")
                return render(request,"teacher.html",{"username":username,"email":email}) 
        messages.success(request,"Success")
    return render(request, "teacher.html")

def import_users(request):
    # Construct the file path
    csv_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'upload_user', 'upload.csv')

    # Check if the file exists
    if not os.path.exists(csv_file_path):
        return render(request, 'error.html', {'message': 'File not found'})

    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            username, email,firstname, password = row

            # Create or update the user
            user, created = User.objects.update_or_create(
                username=username,
                defaults={
                    'email': email,
                    'first_name': firstname,
                }
            )

            # Set the password using the `set_password` method to hash it
            user.set_password(password)
            user.save()

    messages.success(request, 'Users imported successfully.')

    # Redirect to the teacher dashboard
    return redirect('teacher')


def logout(request):
    auth.logout(request)
    return redirect("slogin")
