import cv2
import numpy as np

# Function to calculate vanishing point from two lines defined by points
def calculate_vanishing_point(point1, point2):
    # Unpack line points
    x1, y1 = point1[0]
    x2, y2 = point1[1]
    x3, y3 = point2[0]
    x4, y4 = point2[1]

    # Represent lines in homogeneous form: ax + by + c = 0
    line1_coeffs = np.cross([x1, y1, 1], [x2, y2, 1])
    line2_coeffs = np.cross([x3, y3, 1], [x4, y4, 1])

    # Find the intersection point (vanishing point)
    vp = np.cross(line1_coeffs, line2_coeffs)
    
    # Convert homogeneous coordinates back to Cartesian
    if vp[2] != 0:
        vp = vp / vp[2]
    else:
        return None  # Parallel or coincident lines, no vanishing point
    
    return (vp[0], vp[1])

# Function to check if two points are close (indicating parallel lines)
def is_parallel(vp1, vp2, tolerance=10):
    return np.linalg.norm(np.array(vp1) - np.array(vp2)) < tolerance

# Main function to let the user select lines and compare vanishing points
def main():
    # Load image and set up OpenCV window
    image_path = "WhatsApp.jpg"  # Replace with the path to your image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return
    
    points = []
    def select_point(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Points", image)

    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", select_point)

    # Instruction for the user
    print("Please select 8 points: two points for each line. First two lines should be parallel.")

    while len(points) < 8:
        cv2.imshow("Select Points", image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Split the points into line pairs
    line1 = points[:2]
    line2 = points[2:4]
    test_line1 = points[4:6]
    test_line2 = points[6:8]

    # Calculate vanishing points
    correct_vp = calculate_vanishing_point(line1, line2)
    test_vp = calculate_vanishing_point(test_line1, test_line2)
    print('correct_vp'+str(correct_vp))
    print("test_vp"+str(test_vp))
    if correct_vp and test_vp:
        if is_parallel(correct_vp, test_vp):
            print("The test lines are approximately parallel in 3D.")
        else:
            print("The test lines are not parallel in 3D.")
    else:
        print("Could not calculate one or both vanishing points.")

if __name__ == "__main__":
    main()
