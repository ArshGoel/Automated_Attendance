import os
import cv2
import time
import json
import shutil
import datetime
import numpy as np
import face_recognition
from datetime import date
from django.conf import settings
from django.utils import timezone
from django.contrib import messages
from django.http import JsonResponse
from Services.models import Attendance
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse
# Define the paths for saving images
MEDIA_ROOT = settings.MEDIA_ROOT
PROCESSED_PATH = os.path.join(MEDIA_ROOT, 'processed')
ORIGINAL_PATH = os.path.join(settings.MEDIA_ROOT, 'original')

def load_training_data(directory):
    encodings_list = []
    labels_list = []
    
    for file_name in os.listdir(directory):
        if file_name.endswith('.npz'):
            file_path = os.path.join(directory, file_name)
            if os.path.exists(file_path):
                data = np.load(file_path, allow_pickle=True)
                encodings_list.append(data['encodings'])
                labels_list.append(data['labels'])
    
    if encodings_list and labels_list:
        # Concatenate all encodings and labels
        all_encodings = np.vstack(encodings_list)
        all_labels = np.concatenate(labels_list)
        return list(all_encodings), list(all_labels)
    
    return [], []

def process_images(file_path):
    training_directory = r'D:\PYTHON\Smart Classroom Management Software (SCMS)\Django\SCMS\static\preprocess'
    face_encodings, face_labels = load_training_data(training_directory)
    if not face_encodings or not face_labels:
        return [], None

    test_image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(test_image)

    if not face_locations:
        return [], None

    face_encodings_in_image = face_recognition.face_encodings(test_image, face_locations)
    img_with_boxes = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

    results = []  # This will store the names of the recognized faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings_in_image):
        distances = face_recognition.face_distance(face_encodings, face_encoding)
        best_match_index = np.argmin(distances)
        if distances[best_match_index] < 0.6:
            predicted_name = face_labels[best_match_index]
        else:
            predicted_name = "Unknown"

        results.append(predicted_name)
        cv2.rectangle(img_with_boxes, (left, top), (right, bottom), (0, 255, 0), 2)  # Green rectangle for recognized faces

    # Resize image
    desired_width = 800  # Desired width
    desired_height = 600  # Desired height
    img_with_boxes_resized = cv2.resize(img_with_boxes, (desired_width, desired_height))

    # Ensure the 'processed' directory exists
    processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Save the resized image in MEDIA_ROOT
    processed_image_path = os.path.join(processed_dir, f'processed_{os.path.basename(file_path)}')
    cv2.imwrite(processed_image_path, img_with_boxes_resized)

    # Return the relative URL for use in templates
    relative_path = os.path.relpath(processed_image_path, settings.MEDIA_ROOT)
    return results, relative_path

def teacher(request):
    days = [(datetime.datetime.now() + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    if request.method == 'POST':
        date = request.POST.get('date')
        if date:
            images = request.FILES.getlist('images')
            file_paths = []
            for image in images:
                # Generate a unique filename based on the current datetime for the original image
                original_image_filename = f'original_{datetime.datetime.now().strftime("%d%m%y_%I%m%S")}.jpg'
                original_image_path = os.path.join(ORIGINAL_PATH, original_image_filename)
                # Ensure the 'original' directory exists
                if not os.path.exists(ORIGINAL_PATH):
                    os.makedirs(ORIGINAL_PATH)
                # Save the original image to the original folder
                temp_image_path = default_storage.save(original_image_path, ContentFile(image.read()))
                file_paths.append(temp_image_path)
            results = []
            recognized_students = set()
            for file_path in file_paths:
                full_file_path = os.path.join(settings.MEDIA_ROOT, file_path)
                result, img_with_boxes_path = process_images(full_file_path)
                recognized_students.update(result)
                # Construct URL for processed image
                processed_image_url = os.path.join(settings.MEDIA_URL, 'processed', os.path.basename(img_with_boxes_path)) # type: ignore
                results.append({
                    'result': result,
                    'image_path': processed_image_url
                })
            # Store results and recognized students in session
            request.session['attendance_results'] = results
            request.session['attendance_date'] = date
            request.session['recognized_students'] = list(recognized_students)
            return redirect('attendance')  # Redirect to the attendance view
    return render(request, 'teacher_dashboard.html', {'days': days})


def attendance(request):
    # Retrieve results and recognized students from session
    results = request.session.get('attendance_results', [])
    date = request.session.get('attendance_date', '')
    recognized_students = request.session.get('recognized_students', [])
    # Clear session data after retrieval
    request.session.pop('attendance_results', None)
    request.session.pop('attendance_date', None)
    request.session.pop('recognized_students', None)
    return render(request, 'attendance.html', {
        'results': results,
        'date': date,
        'recognized_students': recognized_students,
    })

@login_required
def save_attendance(request):
    if request.method == 'POST':
        date_str = request.POST.get('date')
        detected_faces_json = request.POST.get('detected_faces')
        
        if not detected_faces_json:
            return JsonResponse({'error': 'Detected faces are required'}, status=400)
        
        try:
            detected_faces = json.loads(detected_faces_json)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid detected faces data'}, status=400)
        
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()

        # Get or create the attendance record for the given date
        attendance_record, created = Attendance.objects.get_or_create(date=date)
        
        # Mark all students as absent initially by creating or updating AttendanceReport
        student_ids_present = []  # List to keep track of present student IDs

        # Mark detected students as present
        for name in detected_faces:
            if name != "Unknown":
                if User.objects.filter(username=name).exists():
                    student_ids_present.append(name)
        
        attendance_record.students_present = student_ids_present
        attendance_record.save()

        # Backup face_data.npz
        face_data_file = r'D:\PYTHON\Smart Classroom Management Software (SCMS)\Django\SCMS\static\preprocess\face_data.npz'
        backup_file = r'D:\PYTHON\Smart Classroom Management Software (SCMS)\Django\SCMS\static\preprocess\face_data_backup.npz'
        if os.path.exists(face_data_file):
            shutil.copy2(face_data_file, backup_file)

        # Load existing data from face_data.npz
        if os.path.exists(face_data_file):
            data = np.load(face_data_file, allow_pickle=True)
            existing_encodings = data['encodings']
            existing_labels = data['labels']
        else:
            existing_encodings = np.empty((0, 128))  # Assuming 128 is the encoding size
            existing_labels = np.array([])

        # Load new data from face_data1.npz
        processed_image_file = r'D:\PYTHON\Smart Classroom Management Software (SCMS)\Django\SCMS\static\preprocess\face_data1.npz'
        if os.path.exists(processed_image_file):
            processed_data = np.load(processed_image_file, allow_pickle=True)
            new_encodings = processed_data['encodings']
            new_labels = processed_data['labels']
        else:
            new_encodings = np.empty((0, 128))  # Assuming 128 is the encoding size
            new_labels = np.array([])

        # Combine old and new data
        if existing_encodings.size and new_encodings.size:
            # Ensure that the combined data retains the best quality
            combined_encodings = np.vstack([existing_encodings, new_encodings])
            combined_labels = np.concatenate([existing_labels, new_labels])
        elif existing_encodings.size:
            combined_encodings = existing_encodings
            combined_labels = existing_labels
        else:
            combined_encodings = new_encodings
            combined_labels = new_labels

        # Save combined data back to face_data.npz
        np.savez(face_data_file, encodings=combined_encodings, labels=combined_labels)

        # Success message and redirect
        messages.success(request, 'Successfully marked Attendance and updated face data')
        return redirect('teacher')

    return JsonResponse({'error': 'Invalid request method'}, status=400)

@login_required
def student(request):
    student = request.user  # Assuming the logged-in user is the student
    today = datetime.date.today()
    period = request.GET.get('period', 'week')  # Default to 'week' if not provided

    if period == 'week':
        week_start = today - datetime.timedelta(days=today.weekday())  # Start of the week (Monday)
        week_end = week_start + datetime.timedelta(days=6)  # End of the week (Sunday)
        start_date, end_date = week_start, week_end
        
        # Prepare attendance data for each day of the week
        attendance_details = []
        for single_date in (week_start + datetime.timedelta(n) for n in range(7)):
            record = Attendance.objects.filter(date=single_date).first()
            if record:
                status = 'Present' if student.username in record.students_present else 'Absent'
            else:
                status = 'Nil'
            attendance_details.append({
                'date': single_date,
                'status': status
            })
        attendance_details.sort(key=lambda x: x['date'])  # Ensure proper date order

        context = {
            'attendance_details': attendance_details,
            'week_start': week_start,
            'week_end': week_end,
            'current_period': period
        }

    elif period == 'month':
        start_date = today.replace(day=1)  # Start of the month
        end_date = (start_date + datetime.timedelta(days=31)).replace(day=1) - datetime.timedelta(days=1)  # End of the month

        # Prepare a calendar view
        calendar_data = []
        current_day = start_date
        while current_day <= end_date:
            record = Attendance.objects.filter(date=current_day).first()
            if record:
                status = 'Present' if student.username in record.students_present else 'Absent'
            else:
                status = 'Nil'
            calendar_data.append({
                'date': current_day,
                'status': status
            })
            current_day += datetime.timedelta(days=1)

        # Organize calendar data into weeks
        weeks = []
        current_week = []
        for day in calendar_data:
            if len(current_week) == 7:
                weeks.append(current_week)
                current_week = []
            current_week.append(day)
        if current_week:
            weeks.append(current_week)
        
        context = {
            'weeks': weeks,
            'start_date': start_date,
            'end_date': end_date,
            'current_period': period
        }

    else:
        context = {
            'attendance_details': [],
            'message': 'Invalid period selected'
        }

    return render(request, 'student_dashboard.html', context)


def video_capture(request):
    training_directory = r'D:\PYTHON\Smart Classroom Management Software (SCMS)\Django\SCMS\static\preprocess'
    encodings, labels = load_training_data(training_directory)
    
    def gen():
        video_capture = cv2.VideoCapture(0)  # Open the default camera
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            if not ret:
                break

            # Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
            rgb_frame = frame[:, :, ::-1]

            # Find all face locations and face encodings in the frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(encodings, face_encoding)
                name = "Unknown"

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = labels[best_match_index]

                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Encode the frame to JPEG and return it as a response
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        video_capture.release()

    return StreamingHttpResponse(gen(), content_type='multipart/x-mixed-replace; boundary=frame')


def video_capture_view(request):
    return render(request, 'video_capture.html')