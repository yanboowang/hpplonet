import os
import numpy as np
import open3d as o3d
from scipy.io import savemat

# Define the data size array
datasize = [4541, 1101, 4661, 801, 271, 2761, 1101, 1101, 4071, 1591, 1201]


# Function to create directory if it does not exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Iterate over each sequence
for seq in range(len(datasize)):
    # Create directory for the current sequence
    output_dir = f'/dataset/data_odometry_velodyne/dataset_zbb/sequences/{seq:02d}/pointground'
    create_dir(output_dir)

    # Iterate over each point cloud file in the sequence
    for bin_idx in range(datasize[seq]):
        # Construct file path for the binary point cloud file
        bin_path = f'/dataset/data_odometry_velodyne/dataset/sequences/{seq:02d}/velodyne/{bin_idx:06d}.bin'

        # Read the point cloud data from the binary file
        point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]

        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        # Segment the ground plane
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.19, ransac_n=3, num_iterations=1000)
        # ground 是 inliers (一维数组)
        ground_indices = np.array(inliers)
        # print("ground_indices shape: ", ground_indices.shape)

        # other 是非 inliers 的索引
        other_indices = np.setdiff1d(np.arange(len(pcd.points)), inliers)
        # print("other_indices shape: ", other_indices.shape)


        # Save the ground and other points to .mat files
        output_filename = f'{bin_idx:06d}.mat'
        output_filepath = os.path.join(output_dir, output_filename)

        # Create dictionaries to save
        save_data = {"ground": ground_indices, "other": other_indices}

        savemat(output_filepath, save_data)
        # print(f'Ground points shape: {ground_points.shape}')
        # print(f'Other points shape: {other_points.shape}')

        print(f'Sequence {seq:02d}, File {bin_idx:06d} processed.')
