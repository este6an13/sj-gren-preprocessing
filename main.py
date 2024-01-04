import cv2
import numpy as np
import h5py
import os

def get_average_neighbors_within_radius(stats, radius):
    # Extract coordinates
    centers = stats[:, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP]] + stats[:, [cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]] // 2

    # Calculate the pairwise distances using vectorized operations
    pairwise_distances = np.linalg.norm(centers[:, np.newaxis, :] - centers, axis=2)

    # Create a binary mask indicating whether the distance is within the specified radius
    neighbors_mask = (pairwise_distances > 0) & (pairwise_distances <= radius)

    # Count the number of neighbors for each component
    num_neighbors = np.sum(neighbors_mask, axis=1)

    # Calculate the average number of neighbors
    average_neighbors = np.mean(num_neighbors) if num_neighbors.size > 0 else 0.0

    return average_neighbors

def get_average_distance(stats):
    # Extract coordinates
    centers = stats[:, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP]] + stats[:, [cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]] // 2

    # Reshape the centers to enable broadcasting
    centers = centers[:, np.newaxis, :]

    # Calculate the pairwise distances using vectorized operations
    pairwise_distances = np.linalg.norm(centers - centers.transpose((1, 0, 2)), axis=2)

    # Mask diagonal values (distance to itself) and flatten the distances
    distances = pairwise_distances[np.triu(np.ones_like(pairwise_distances), k=1) == 1]

    # Calculate the average distance per point
    average_distance_per_point = np.sum(distances) / len(distances) if len(distances) > 0 else 0.0

    return average_distance_per_point


def read_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            images.append(image)
    return images

def get_connected_components_features(images):
    # Initialize a list to store the counts
    features = []

    # Iterate through each image in the tensor
    for i in range(len(images)):
        print(i+1, len(images))
        resized = cv2.resize(images[i], (512, 512))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Convert torch tensor to NumPy array
        mask = (gray < 250).astype(np.uint8)
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        equalized = cv2.equalizeHist(masked)

        binary = cv2.inRange(equalized, 1, 10)

        # Apply erosion to make dark points thinner
        #kernel = np.ones((3, 3), np.uint8)
        #eroded = cv2.erode(binary, kernel, iterations=2)

        # Perform connected component analysis
        # Separate the two colors (e.g., 0 and 255)
        color_255 = (binary == 255).astype(np.uint8)  # Binary image for color 255

        # Apply connected components analysis separately
        _, _, stats_255, _ = cv2.connectedComponentsWithStats(color_255, connectivity=8)

        # Count the number of components for each color
        dark_points_count = len(stats_255) - 1  # Exclude the background
        average_neighbors = get_average_neighbors_within_radius(stats_255, 50.0)
        average_distance = get_average_distance(stats_255)

        # Append the count to the list
        features.append([dark_points_count, average_neighbors, average_distance])

    features = np.array(features)
    return features

def write_to_h5(images, output_file):
    with h5py.File(output_file, 'w') as hf:
        for i in range(len(images)):
            hf.create_dataset(f'image_{i}', data=images[i])

def main():
    syndrome_folder = r'D:\dequi\Documents\REDN-Project\syndrome_images_200'
    non_syndrome_folder = r'D:\dequi\Documents\REDN-Project\non_syndrome_images_200'

    syndrome_images = read_images(syndrome_folder)
    non_syndrome_images = read_images(non_syndrome_folder)

    processed_syndrome = get_connected_components_features(syndrome_images)
    processed_non_syndrome = get_connected_components_features(non_syndrome_images)

    write_to_h5(processed_syndrome, 'syndrome_features_200.h5')
    write_to_h5(processed_non_syndrome, 'non_syndrome_features_200.h5')

if __name__ == "__main__":
    main()
