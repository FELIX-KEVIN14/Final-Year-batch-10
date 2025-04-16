from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# Function to generate a unique color for each class
def generate_color_map(num_colors):
    np.random.seed(0)  # For reproducibility
    colors = np.random.randint(0, 255, size=(num_colors, 3), dtype=int)
    return colors

# Load YOLOv8 model
model = YOLO('best_model.pt')  # Path to your YOLOv8 model

# Load the video
video_path = r"/Users/felixkevinsinght/construction/ppe.mp4"  # Path to your video file
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = 0

# Prepare for CSV output
timestamp = datetime.now().strftime('%d_%m_%Y')
csv_filename = f'ppe_prediction_{timestamp}.csv'
csv_data = []

# Initialize counters and lists
last_record_time = 0

# Generate unique colors for each class
num_classes = len(model.names)
colors = generate_color_map(num_classes)

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    current_time = frame_count / fps

    # Perform inference
    results = model(frame)
    boxes = results[0].boxes

    # Reset counts
    total_people = 0
    distinct_classes = set()
    class_counts = defaultdict(int)

    # Draw the bounding boxes on the frame
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # xyxy format for box coordinates
        conf = box.conf[0]  # Confidence score
        cls = int(box.cls[0])  # Class label

        total_people += 1
        distinct_classes.add(cls)
        class_counts[cls] += 1

        # Ensure color is in valid format (B, G, R)
        if cls == 0:
            color = (0, 0, 255)  # Red for person
        else:
            color = tuple(colors[cls])  # Unique color for other classes
        
        # Ensure color values are integers and in range [0, 255]
        color = tuple(map(lambda x: min(max(int(x), 0), 255), color))

        # Draw a rectangle around the detected object
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Put label text (class name and confidence score)
        label = f'{model.names[cls]} {conf:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Store data for CSV every 2 seconds
    if int(current_time) % 2 == 0 and int(current_time) != last_record_time:
        last_record_time = int(current_time)
        csv_data.append({
            'Time (s)': int(current_time),
            'Hour-Min-Sec': datetime.now().strftime('%H:%M:%S'),
            'Total People': total_people,
            'Distinct Classes': len(distinct_classes),
            **{f'Class {model.names[cls]}': count for cls, count in class_counts.items()}
        })

    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Save the CSV file
df = pd.DataFrame(csv_data)
df.to_csv(csv_filename, index=False)

print(f'CSV file saved as {csv_filename}')
