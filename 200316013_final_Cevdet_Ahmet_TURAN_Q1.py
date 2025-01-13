import cv2
import numpy as np
import csv 

dataset = []

def save_dataset_to_csv(dataset, filename="CATs_output_dataset.csv"):
    headers = ["X Coordinate", "Y Coordinate", "Pattern Type"]
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)  
        writer.writerows(dataset)  
    print(f"Dataset saved to {filename}")


def mouse_click(event, x, y, flags, param):
    global dataset
    if event == cv2.EVENT_LBUTTONDOWN:
        region_size = 30
        x_min = max(x - region_size, 0)
        y_min = max(y - region_size, 0)
        x_max = min(x + region_size, param.shape[1] - 1)
        y_max = min(y + region_size, param.shape[0] - 1)

        pattern_region = param[y_min:y_max, x_min:x_max]
        gray = cv2.cvtColor(pattern_region, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=10,
            param1=50,
            param2=15,
            minRadius=5,
            maxRadius=30,
        )

        pattern_type = "Unknown"
        if circles is not None:
            pattern_type = "Circle" 
        else:
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) >= 8:  
                    pattern_type = "Cross Sign"
                    break

        dataset.append((x, y, pattern_type))
        print(f"Added: ({x}, {y}, {pattern_type})")
def capture_image():
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Error: Cannot access the camera")
        return None

    print("Press 's' to capture the image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("CAT Feed", frame)

        # Press 's' to capture and save the image
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("Image captured.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame

def display_and_capture_clicks(image):
    if image is None:
        print("No image to process.")
        return

    print("Click on the center of the circles and cross signs in the image.")
    cv2.imshow("Captured Image", image)

    cv2.setMouseCallback("Captured Image", mouse_click, param=image)

    while True:
        cv2.imshow("Captured Image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting click capture mode.")
            break

    cv2.destroyAllWindows()

def main():
    image = capture_image()

    display_and_capture_clicks(image)

    print("\nSaving dataset..")
    save_dataset_to_csv(dataset)

    print("\nFinal Dataset:")
    for record in dataset:
        print(record)

if __name__ == "__main__":
    main()
