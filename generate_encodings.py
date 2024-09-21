import face_recognition
import pickle
import os

# Directory containing known images
known_images_dir = r'C:\Users\Welcome\OneDrive\Desktop\known_images'

# Lists to store encodings and names
encodeListKnown = []
classNames = []

# Iterate through the images in the directory
for filename in os.listdir(known_images_dir):
    if filename.endswith('.jpeg') or filename.endswith('.png'):
        image_path = os.path.join(known_images_dir, filename)
        img = face_recognition.load_image_file(image_path)
        encode = face_recognition.face_encodings(img)[0]
        
        # Extract name from filename (assuming format "name.jpg")
        name = os.path.splitext(filename)[0]
        encodeListKnown.append(encode)
        classNames.append(name)

# Save encodings and names to a pickle file
with open('encodings.pkl', 'wb') as f:
    pickle.dump([encodeListKnown, classNames], f)

print('Encodings saved to encodings.pkl')
