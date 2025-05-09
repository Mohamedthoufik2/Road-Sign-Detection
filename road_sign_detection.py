import cv2
import numpy as np
import os

# Load template images of signs from 'templates/' directory
def load_templates(template_dir='templates'):
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith(('.png', '.jpg')):
            img = cv2.imread(os.path.join(template_dir, filename), 0)
            templates[filename.split('.')[0]] = img
    return templates

# Match templates to input frame
def detect_signs(frame, templates, threshold=0.7):
    detections = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for name, template in templates.items():
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            detections.append((name, pt, (pt[0] + w, pt[1] + h)))
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (pt[0], pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)
    return frame, detections

# Main function
def main():
    templates = load_templates()
    if not templates:
        print("No templates found. Add road sign images to the 'templates/' folder.")
        return

    cap = cv2.VideoCapture(0)  # Use webcam

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, detections = detect_signs(frame, templates)

        cv2.imshow('Road Sign Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
