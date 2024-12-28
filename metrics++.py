import numpy as np
import sqlite3
import cv2
import matplotlib.pyplot as plt

def draw_epipolar_lines(img1, img2, lines, pts1, pts2):
    """
    Draw epipolar lines on the images.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.
        lines (np.ndarray): Epipolar lines.
        pts1 (np.ndarray): Points in the first image.
        pts2 (np.ndarray): Points in the second image.

    Returns:
        np.ndarray: Images with epipolar lines and points drawn.
    """
    r, c = img1.shape[:2]
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img1_color = cv2.circle(img1_color, tuple(map(int, pt1)), 5, color, -1)
        img2_color = cv2.circle(img2_color, tuple(map(int, pt2)), 5, color, -1)

    return img1_color, img2_color

def visualize_matches_and_epilines(img1_path, img2_path, match_points1, match_points2, fundamental_matrix):
    """
    Visualize matches and epipolar lines.

    Args:
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.
        match_points1 (np.ndarray): Matching points in the first image.
        match_points2 (np.ndarray): Matching points in the second image.
        fundamental_matrix (np.ndarray): Fundamental matrix.

    Returns:
        None
    """
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both image paths are incorrect or the images cannot be read.")

    # Compute epilines for points in the second image and draw them on the first image
    lines1 = cv2.computeCorrespondEpilines(match_points2.reshape(-1, 1, 2), 2, fundamental_matrix)
    lines1 = lines1.reshape(-1, 3)
    img1_epilines, img2_points = draw_epipolar_lines(img1, img2, lines1, match_points1, match_points2)

    # Compute epilines for points in the first image and draw them on the second image
    lines2 = cv2.computeCorrespondEpilines(match_points1.reshape(-1, 1, 2), 1, fundamental_matrix)
    lines2 = lines2.reshape(-1, 3)
    img2_epilines, img1_points = draw_epipolar_lines(img2, img1, lines2, match_points2, match_points1)

    # Display the images with epipolar lines and points
    plt.figure(figsize=(15, 5))
    plt.subplot(121), plt.imshow(img1_epilines)
    plt.title('Epipolar Lines in Image 1'), plt.axis('off')
    plt.subplot(122), plt.imshow(img2_epilines)
    plt.title('Epipolar Lines in Image 2'), plt.axis('off')
    plt.show()

def extract_match_points(database_path, image_id1, image_id2):
    """
    Extract matching keypoints for two images from COLMAP database.

    Args:
        database_path (str): Path to COLMAP database.db.
        image_id1 (int): ID of the first image.
        image_id2 (int): ID of the second image.

    Returns:
        match_points1 (np.ndarray): Matching points in the first image.
        match_points2 (np.ndarray): Matching points in the second image.
    """
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Query matches table for the pair of images
    pair_id = image_pair_id(image_id1, image_id2)
    cursor.execute("SELECT data FROM matches WHERE pair_id = ?", (pair_id,))
    result = cursor.fetchone()

    if result is None:
        raise ValueError("No matches found between the specified images.")

    matches = np.frombuffer(result[0], dtype=np.uint32).reshape(-1, 2)

    # Extract keypoints for both images
    match_points1 = extract_keypoints(cursor, image_id1, matches[:, 0])
    match_points2 = extract_keypoints(cursor, image_id2, matches[:, 1])

    conn.close()
    return match_points1, match_points2

def extract_keypoints(cursor, image_id, indices):
    """
    Extract keypoint coordinates for a specific image and indices.

    Args:
        cursor (sqlite3.Cursor): SQLite cursor.
        image_id (int): Image ID.
        indices (list): List of keypoint indices.

    Returns:
        np.ndarray: Keypoint coordinates.
    """
    cursor.execute("SELECT data FROM keypoints WHERE image_id = ?", (image_id,))
    keypoints_data = cursor.fetchone()[0]
    keypoints = np.frombuffer(keypoints_data, dtype=np.float32).reshape(-1, 6)

    # Filter out indices that are out of bounds
    valid_indices = indices[indices < keypoints.shape[0]]

    return keypoints[valid_indices, :2]

def image_pair_id(image_id1, image_id2):
    """
    Compute pair_id for two image IDs.

    Args:
        image_id1 (int): ID of the first image.
        image_id2 (int): ID of the second image.

    Returns:
        int: Pair ID.
    """
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * 2147483647 + image_id2

def compute_fundamental_matrix(match_points1, match_points2):
    """
    Compute the fundamental matrix using RANSAC.

    Args:
        match_points1 (np.ndarray): Matching points in the first image.
        match_points2 (np.ndarray): Matching points in the second image.

    Returns:
        np.ndarray: Fundamental matrix.
    """
    fundamental_matrix, _ = cv2.findFundamentalMat(match_points1, match_points2, cv2.FM_RANSAC)
    return fundamental_matrix

def compute_transformation_matrix(images_txt_path, image_name1, image_name2):
    """
    Compute the transformation matrix (T) between two images.

    Args:
        images_txt_path (str): Path to images.txt from COLMAP.
        image_name1 (str): Name of the first image.
        image_name2 (str): Name of the second image.

    Returns:
        np.ndarray: Transformation matrix T = [R | t].
    """
    poses = {}
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()

    for i in range(4, len(lines), 2):
        data = lines[i].split()
        image_name = data[-1]
        qvec = np.array(data[1:5], dtype=float)  # Quaternion
        tvec = np.array(data[5:8], dtype=float)  # Translation
        poses[image_name] = (qvec, tvec)

    qvec1, tvec1 = poses[image_name1]
    qvec2, tvec2 = poses[image_name2]

    # Convert quaternion to rotation matrix
    R1 = qvec_to_rotmat(qvec1)
    R2 = qvec_to_rotmat(qvec2)

    # Compute relative rotation and translation
    R = R2 @ R1.T
    t = tvec2 - R @ tvec1

    T = np.hstack((R, t.reshape(-1, 1)))
    return T

def qvec_to_rotmat(qvec):
    """
    Convert a quaternion to a rotation matrix.

    Args:
        qvec (np.ndarray): Quaternion (4,).

    Returns:
        np.ndarray: Rotation matrix (3x3).
    """
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])

def calculate_number_of_matches_weight(matches):
    """
    Calculate the weight based on the number of matches between two images.

    Parameters:
        matches (int): Total number of matches between the two images.

    Returns:
        float: Weight calculated as 1 / number of matches.
    """
    if matches <= 0:
        raise ValueError("Number of matches must be greater than zero.")
    return 1 / matches

def calculate_epipolar_error_weight(match_points1, match_points2, fundamental_matrix):
    """
    Calculate the weight based on the average distance of matches from the epipolar lines.

    Parameters:
        match_points1 (ndarray): Coordinates of keypoints in the first image (Nx2).
        match_points2 (ndarray): Coordinates of corresponding keypoints in the second image (Nx2).
        fundamental_matrix (ndarray): The 3x3 fundamental matrix relating the two images.

    Returns:
        float: Weight calculated as the average distance of matches from epipolar lines.
    """
    if len(match_points1) != len(match_points2):
        raise ValueError("Number of points in match_points1 and match_points2 must be the same.")

    num_matches = len(match_points1)
    if num_matches == 0:
        raise ValueError("No matches provided.")

    epilines2 = np.dot(fundamental_matrix, np.c_[match_points1, np.ones(num_matches)].T).T
    epilines1 = np.dot(fundamental_matrix.T, np.c_[match_points2, np.ones(num_matches)].T).T

    epilines2 /= np.linalg.norm(epilines2[:, :2], axis=1, keepdims=True)
    epilines1 /= np.linalg.norm(epilines1[:, :2], axis=1, keepdims=True)

    dist1 = np.abs(np.sum(epilines2 * np.c_[match_points2, np.ones(num_matches)], axis=1))
    dist2 = np.abs(np.sum(epilines1 * np.c_[match_points1, np.ones(num_matches)], axis=1))

    avg_distance = (np.mean(dist1) + np.mean(dist2)) / 2
    return avg_distance

if __name__ == "__main__":
    # Database and file paths
    database_path = r"C:\Users\hp\OneDrive\Desktop\CVG\new_db.db"
    images_txt_path = r"C:\Users\hp\OneDrive\Desktop\CVG\images.txt"
    img1_path = r"C:\Users\hp\OneDrive\Desktop\CVG\cup\images\0000.jpg"
    img2_path = r"C:\Users\hp\OneDrive\Desktop\CVG\cup\images\0032.jpg"  # Fixed file path

    # Input values
    image_id1 = 1
    image_id2 = 33
    image_name1 = "0000.jpg"
    image_name2 = "0032.jpg"

    # Extract match points
    match_points1, match_points2 = extract_match_points(database_path, image_id1, image_id2)

    # Compute fundamental matrix
    fundamental_matrix = compute_fundamental_matrix(match_points1, match_points2)

    # Compute transformation matrix
    T = compute_transformation_matrix(images_txt_path, image_name1, image_name2)

    # Calculate weights
    matches_weight = calculate_number_of_matches_weight(len(match_points1))
    epipolar_error_weight = calculate_epipolar_error_weight(match_points1, match_points2, fundamental_matrix)
    
    # Visualize matches and epipolar lines
    visualize_matches_and_epilines(img1_path, img2_path, match_points1, match_points2, fundamental_matrix)

    # Output results
    print("Fundamental Matrix:\n", fundamental_matrix)
    print("Transformation Matrix (T):\n", T)
    print("Number of Matches Weight:", matches_weight)
    print("Epipolar Error Weight:", epipolar_error_weight)

