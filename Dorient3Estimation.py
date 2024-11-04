import cv2
import numpy as np
import imageio
import math  as math
from transform import transform
import matplotlib.pyplot as plt
# Global points list to store selected points
points = []  # Initialize global points list

import numpy as np
import cv2

def project_3d_to_2d(point_3d, intrinsic_matrix, extrinsic_matrix):
    # Convert the 3D point to homogeneous coordinates
    point_3d_homogeneous = np.array([point_3d[0], point_3d[1], point_3d[2], 1]).reshape(4, 1)
    
    # Project the 3D point into the camera coordinate system using the extrinsic matrix
    camera_coordinates = np.dot(extrinsic_matrix, point_3d_homogeneous)  # 3x1 result
    
    # Project into the image plane using the intrinsic matrix
    image_coordinates_homogeneous = np.dot(intrinsic_matrix, camera_coordinates[:3])
    
    # Convert homogeneous coordinates to 2D by dividing by the z component
    x = image_coordinates_homogeneous[0] / image_coordinates_homogeneous[2]
    y = image_coordinates_homogeneous[1] / image_coordinates_homogeneous[2]
    
    return (int(x), int(y))  # Return as pixel coordinates

# Test the function with known 3D points and check if the output 2D points align
# Example usage (replace with your values):


def putBanner(image,img_points,bannerdegree, banner):
    K = np.array([
    [1126.00516, 0.0, 1006.32321],
    [0.0, 278.159008, 588.130689],
    [0.0, 0.0, 1.0]
    ])
    obj_points = np.array([[0, 0, 0], [0, 12, 0], [6, 12, 0], [6, 0, 0]], dtype=np.float32)
    
    dist_coeffs = np.zeros((4, 1))
    
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeffs)

    if success:
        print("solvePnP succeeded.")
        # Convert rvec to rotation matrix (optional, for extrinsic matrix)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        #print("Extrinsic rvec:\n", rvec)
        extrinsic_matrix = np.hstack((rotation_matrix, tvec))
        #print("Extrinsic Matrix:\n", extrinsic_matrix)
    else:
        print("Error: solvePnP failed to find a solution.")


    cornersBanner3D=np.array(
        [
            [0,0,0],
            [-3* math.cos(math.radians(bannerdegree)),0,3* math.sin(math.radians(bannerdegree))],
            [0.2-3* math.cos(math.radians(bannerdegree)),6,3* math.sin(math.radians(bannerdegree))],
            [0,6,0]
        ]
    )
    cornersIn2D=[]
    for i in cornersBanner3D:
        #print(i)
        projected_point = project_3d_to_2d(i, K, extrinsic_matrix)
        #print(projected_point)
        cornersIn2D.append(projected_point)  # Append the point as a tuple or list
        cv2.circle(image, (projected_point[0], projected_point[1]), 5, (255, 0, 0), -1)
        

    # Convert to a numpy array with shape [4, 2]
    cornersIn2D = np.array(cornersIn2D, dtype=int)
    print("got my corners")
    new=transform(banner,image,cornersIn2D)
    print("Transfromed that bitch")
    # cv2.imshow('Banner',new)
    # cv2.waitKey(10)
    # cv2.imshow("corners",image)
    # cv2.waitKey(10)
    return cornersIn2D,new
    




# Define the mouse callback function
def select_point(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return  # Ignore all other events
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))  # Add the selected point
        # Draw a small circle at each selected point for visual feedback
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Points", image)  # Update image with selected point
        if len(points) == 4:  # Close window when 4 points are selected
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load image and check if it's loaded successfully
    #image = cv2.imread("ImageTest/IMG_9020.JPG")
    #if image is None:
     #   print("Error: Could not load image. Check the file path.")
      #  exit()

    video_path = "clips/clip1.mp4"
    cap = cv2.VideoCapture(video_path)
    banner=cv2.imread('banner.png')
    if not cap.isOpened():
        print("Error: Could not open video.")
        reader = imageio.get_reader(video_path)
        frame = reader.get_data(0)
    else:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video.")
            cap.release()
        cap.release()

    image=frame
    # Set up the window and callback for point selection
    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", select_point)

    # Keep the window open until 4 points are selected
    while len(points) < 4:
        cv2.imshow("Select Points", image)
        cv2.waitKey(1)
    print("Selected points:", points)
    #  camera intrinsic matrix 


    
    # Define known 3D points in object space
    obj_points = np.array([[0, 0, 0], [0, 12, 0], [6, 12, 0], [6, 0, 0]], dtype=np.float32)

    
    dist_coeffs = np.zeros((4, 1))

    # Estimate the extrinsic parameters (rotation and translation vectors)
    
    img_points = np.array(points, dtype=np.float32)
    

    
    #cornersBanner2D=putBanner(image,0,K,extrinsic_matrix)

    # Define video parameters
    output_file = 'putBanner_different_angle.mp4'  # Filename for the output video
    frame_width, frame_height = 1000, 1000  # Set the width and height of the frames
    fps = 1  # Frames per second

    # Initialize the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    # Assume frames is a list of numpy arrays, each representing a frame

    figBanner, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Write each frame to the video file
    for i in range(7):  # Starts from 0 and goes up to 10
        imagetem=np.copy(image)
        cornersBanner2D,imagetem=putBanner(imagetem,img_points,i*15,banner)
        filename = f"banner_{i*15}degree.jpg"
        if i<1:
            cv2.imwrite(filename, imagetem[0:1000,500:-1])
            out.write(imagetem[0:1000,500:-1])
            imagetem = cv2.cvtColor(imagetem, cv2.COLOR_BGR2RGB)
            axes[math.floor(i/3), int(i%3)].imshow(imagetem[0:500,700:1300])
            axes[math.floor(i/3), int(i%3)].set_title(f"{i*15} degree")
            axes[math.floor(i/3), int(i%3)].axis("off")  # Hide axis
        elif i>1:
            cv2.imwrite(filename, imagetem[0:1000,500:-1])
            out.write(imagetem[0:1000,500:-1])
            imagetem = cv2.cvtColor(imagetem, cv2.COLOR_BGR2RGB)
            ii=i-1
            axes[math.floor(ii/3), int(ii%3)].imshow(imagetem[0:500,700:1300])
            axes[math.floor(ii/3), int(ii%3)].set_title(f"{i*15} degree")
            axes[math.floor(ii/3), int(ii%3)].axis("off")  # Hide axis       
    # Release the video writer object
    out.release()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Smaller values for tighter fit
    plt.show()
    print("Video saved as", output_file)
            

