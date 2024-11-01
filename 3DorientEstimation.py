import cv2
import numpy as np
import imageio
import math  as math
from transform import transform
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


def putBanner(image,bannerdegree,K,extrinsic_matrix):

    cornersBanner3D=np.array(
        [
            [0,0,0],
            [0,-20* math.cos(math.radians(bannerdegree)),-20* math.sin(math.radians(bannerdegree))],
            [50,-20* math.cos(math.radians(bannerdegree)),-20* math.sin(math.radians(bannerdegree))],
            [50,0,0]
        ]
    )
    cornersIn2D=[]
    for i in cornersBanner3D:
        print(i)
        projected_point = project_3d_to_2d(i, K, extrinsic_matrix)
        print(projected_point)
        cornersIn2D.append(projected_point)  # Append the point as a tuple or list
        cv2.circle(image, (projected_point[0], projected_point[1]), 5, (255, 0, 0), -1)
        
    cv2.imshow("corners",image)
    cv2.waitKey(5000)
    # Convert to a numpy array with shape [4, 2]
    cornersIn2D = np.array(cornersIn2D, dtype=int)
    new=transform(banner,image,cornersIn2D)
    cv2.imshow('Banner',new)
    cv2.waitKey(0)
    cv2.imshow("corners",image)
    cv2.waitKey(5000)
    return cornersIn2D
    




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

    K = np.array([
    [1126.00516, 0.0, 1006.32321],
    [0.0, 278.159008, 588.130689],
    [0.0, 0.0, 1.0]
    ])
    # Define known 3D points in object space
    obj_points = np.array([[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]], dtype=np.float32)

    # Use selected 2D image points
    img_points = np.array(points, dtype=np.float32)

    # Define distortion coefficients (if available, otherwise use zeros)
    dist_coeffs = np.zeros((4, 1))

    # Estimate the extrinsic parameters (rotation and translation vectors)
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeffs)

    if success:
        # Convert rvec to rotation matrix (optional, for extrinsic matrix)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        print("Extrinsic rvec:\n", rvec)
        extrinsic_matrix = np.hstack((rotation_matrix, tvec))
        print("Extrinsic Matrix:\n", extrinsic_matrix)
    else:
        print("Error: solvePnP failed to find a solution.")
    

    
    cornersBanner2D=putBanner(image,0,K,extrinsic_matrix)

    
        
    #print(cornersBanner2D)
    '''
    # Define a 3D point in world coordinates
    test_3d_point = (0, 0, 0)  # Replace with actual test point

    # Use the intrinsic matrix and extrinsic matrix calculated earlier
    projected_2d_point = project_3d_to_2d(test_3d_point, K, extrinsic_matrix)
    print("Projected 2D point:", projected_2d_point)
    orthognal_3d_point_45=(0,-70,-70)
    orthognal_3d_point_30=(0,-100*math.cos(math.radians(30)),-100*math.sin(math.radians(30)))
    orthognal_3d_point_90=(0,0,-100)
    orthognal_2d_point_30= project_3d_to_2d(orthognal_3d_point_30, K, extrinsic_matrix)
    orthognal_2d_point_45= project_3d_to_2d(orthognal_3d_point_45, K, extrinsic_matrix)
    orthognal_2d_point_90= project_3d_to_2d(orthognal_3d_point_90, K, extrinsic_matrix)
    print("orthognal_2d_point_30:", orthognal_2d_point_30)
    print("orthognal_2d_point_45:", orthognal_2d_point_45)
    print("orthognal_2d_point_90:", orthognal_2d_point_90)
    color30 = (0, 255, 0)  # Green color
    thickness = 5
    color45 = (0, 0, 255)  
    thickness = 5
    color90= (255, 0, 0)  
    thickness = 5
    # Draw the line
    
    cv2.line(image, projected_2d_point, orthognal_2d_point_30, color30, thickness)
    cv2.line(image, projected_2d_point, orthognal_2d_point_45, color45, thickness)
    cv2.line(image, projected_2d_point, orthognal_2d_point_90, color90, thickness)
    cv2.imshow('Orthognal',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
