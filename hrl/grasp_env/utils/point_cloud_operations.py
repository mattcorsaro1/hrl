'''
Copyright, 2021, Matt Corsaro, matthew_corsaro@brown.edu
'''

import numpy as np
import open3d as o3d

# Copied from cloud_loader
def cropSingleCloud(cloud):
    return cloud[np.logical_and(
                np.logical_and(\
                np.logical_and(\
                    np.abs(cloud[:,0]) < 0.09, \
                    np.abs(cloud[:,1]) < 0.076), \
                    cloud[:,2] < 0.155), \
                cloud[:,2] > -0.085)]
    # z - 1 cm behind sampled pt plus 0.075 for precision

def sampleCloud(cloud, desired_num_points):
    # Upsample if too few points
    if cloud.shape[0] < desired_num_points:
        return upsampleCloud(cloud, desired_num_points)
    elif cloud.shape[0] > desired_num_points:
        return downsampleCloud(cloud, desired_num_points)
    else:
        return cloud

def downsampleCloud(cloud, desired_num_points):
    assert(cloud.shape[0] > desired_num_points)
    valid_indices = np.arange(cloud.shape[0])
    np.random.shuffle(valid_indices)
    downsampled_cloud = cloud[valid_indices[:desired_num_points], :]
    assert(downsampled_cloud.shape[0] == desired_num_points)
    return downsampled_cloud

def upsampleCloud(cloud, desired_num_points, noise=0.003):
    assert(cloud.shape[0] < desired_num_points)
    num_additional_points = desired_num_points - cloud.shape[0]
    new_points = []
    for i in range(num_additional_points):
        new_points.append(cloud[np.random.choice(cloud.shape[0]), :] + (np.random.random((1, 3))-0.5)*noise)
    new_point_cloud = np.array((new_points)).reshape((-1, 3))
    upsampled_cloud = np.concatenate((cloud, new_point_cloud))
    assert(upsampled_cloud.shape[0] == desired_num_points)
    return upsampled_cloud
