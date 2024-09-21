import os
import cv2
import face_recognition
import pickle
from datetime import datetime, timedelta
import time

# Load encodings and class names from the file
with open('encodings.pkl', 'rb') as f:
    encodeListKnown, classNames = pickle.load(f)

print('Encodings Loaded.')

# Function to mark attendance in a session-specific CSV file
def markAttendance(name, subject, overall_status):
    filename = f'{subject}_Attendance.csv'

    # Error handling for file permission issues
    try:
        file_exists = os.path.isfile(filename)

        # Read the file if it exists
        if file_exists:
            with open(filename, 'r') as f:
                myDataList = f.readlines()
        else:
            myDataList = []

        nameList = [line.split(',')[0] for line in myDataList]

        # Check if the person is already marked for this session
        for line in myDataList:
            entry = line.split(',')
            if entry[0] == name and overall_status in entry:
                return  # Already marked

        # Mark attendance
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')

        # Append the attendance to the file
        with open(filename, 'a') as f:
            if not file_exists:
                f.write('Name,Time,Overall Status\n')  # Write header if file is created
            f.write(f'{name},{dtString},{overall_status}\n')

    except PermissionError:
        print(f"Permission denied: Could not write to {filename}. Check if the file is open elsewhere.")

# Function to process attendance for a single detection
def processAttendance(window_time_str, session, subject, attendance_dict):
    window_time = datetime.strptime(window_time_str, "%I:%M %p").time()
    time_window_start = (datetime.combine(datetime.today(), window_time)).time()
    time_window_end = (datetime.combine(datetime.today(), window_time) + timedelta(minutes=1)).time()

    while True:
        now = datetime.now().time()

        # Wait until the specified time window to activate the webcam
        if time_window_start <= now <= time_window_end:
            cap = cv2.VideoCapture(0)
            print(f"{session.capitalize()} time detected.")

            success, img = cap.read()
            if not success:
                print("Failed to capture image from webcam. Please check the camera.")
                break

            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = faceDis.argmin()

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()  # Ensure name is in uppercase
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                    # Mark attendance for the session
                    attendance_dict[name][session] = True
                    print(f"{session.capitalize()} time attendance marked for {name}. Exiting...")
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Release resources in case of failure to detect
            cap.release()
            cv2.destroyAllWindows()

        # Sleep briefly before rechecking the time
        time.sleep(1)

# Main loop to process both start and end times
def main():
    subject = "english"  # Example subject
    attendance_dict = {name.upper(): {'start': False, 'end': False} for name in classNames}  # Tracking attendance

    print("Waiting for start time...")
    # Process start time attendance
    processAttendance("10:27 PM", "start", subject, attendance_dict)

    print("Waiting for end time...")
    # Wait for the end time and process end time attendance
    processAttendance("10:28 PM", "end", subject, attendance_dict)

    # After both sessions, determine overall status (Present/Absent)
    for name, sessions in attendance_dict.items():
        if sessions['start'] and sessions['end']:
            overall_status = 'Present'
        else:
            overall_status = 'Absent'

        # Mark the overall status in the CSV
        markAttendance(name, subject, overall_status)

if __name__ == "__main__":
    main()
