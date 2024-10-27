import cv2
import numpy as np
import imageio

def get_in_cal(frame):
    imageSize = (frame.shape[1], frame.shape[0])
    world_points = np.array([
    [0, 0, 0],       # Bottom-left corner of the goalpost in real-world
    [7.32, 0, 0],    # Bottom-right corner of the goalpost (7.32 meters width of the goal)
    [0,2.44,0],      # Top left corner of the goalpost
    [7.32, 2.44, 0], # Top right corner of the goalpost
    [-5.5, 5.5, 0],    # Penalty box corner left
    [12.82, 5.5, 0],
    [-16.5, 16.5, 0],    #bigger square left
    [28.82, 16.5, 0],
    ], dtype=np.float32)

#for frame 1 of clip1 done manually for now, use the function get_image_points to reselect
    image_points = np.array([
    (1545, 373),    
    (1756, 432),    
    (1553, 265),    
    (1763, 315),    
    (1227, 354),    
    (1706, 505),    
    (629, 316),     
    (1566, 711),    
], dtype=np.float32)

    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera([world_points], [image_points], imageSize, None, None)

    return cameraMatrix, distCoeffs, rvecs, tvecs


#helper function, prints the clicked coordinate so i can copy that stuff
def select_point(event, x, y, flags, param):
    global image_points
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add the point to the list
        image_points.append((x, y))
        # Show the point on the image
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Points", param)
        print(f"Point selected: ({x}, {y})")

#function to open image and select points manually as the line detection thing is not really working
def get_image_points(frame):
    global image_points
    image_points = []

    cv2.imshow("Select Points", frame)
    cv2.setMouseCallback("Select Points", select_point, frame)
    
    print("Click on the image to select points. Press 'q' to finish.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    return image_points

if __name__ == "__main__":
    video_path = "./clips/clip2.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        reader = imageio.get_reader('./clips/clip2.mp4')
        frame = reader.get_data(0)
    else:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video.")
            cap.release()
    
    """ to recalibrate uncomment this :) """
    points = get_image_points(frame)
    print("Selected Image Points:", points)

    print(get_in_cal(frame))
