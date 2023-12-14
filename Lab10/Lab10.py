import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# List of video paths
video_paths = ["traffic3.mp4", "traffic.mp4"]
output_paths = ["output_traffic3.mp4", "output_traffic.mp4"]

for i, video_path in enumerate(video_paths):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Adjust the codec based on your preference
    output_path = output_paths[i]
    output_video = cv2.VideoWriter(output_path, fourcc, 25.0, (int(cap.get(3)), int(cap.get(4))))

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Write the frame to the output video
            output_video.write(annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and the VideoWriter
    cap.release()
    output_video.release()

    # Close the display window 
    cv2.destroyAllWindows() 
    print(f"Output video saved to: {output_path}")