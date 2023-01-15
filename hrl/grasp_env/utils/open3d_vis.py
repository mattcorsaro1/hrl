'''
Copyright, 2021, Matt Corsaro, matthew_corsaro@brown.edu
'''

import numpy as np
import open3d as o3d

# Given a 4x4 transformation matrix, create coordinate frame mesh at the pose
#     and scale down.
def o3dTFAtPose(pose, scale_down=10):
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    scaling_maxtrix = np.ones((4,4))
    scaling_maxtrix[:3, :3] = scaling_maxtrix[:3, :3]/scale_down
    scaled_pose = pose*scaling_maxtrix
    axes.transform(scaled_pose)
    return axes

def visualizeGraspPose(cloud_points, pose=None):
    if type(cloud_points) == o3d.geometry.PointCloud:
        cloud = cloud_points
    else:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_points)
    world_axis = o3dTFAtPose(np.eye(4))
    models = [cloud, world_axis]
    if pose is not None:
        models.append(o3dTFAtPose(pose, scale_down=100))
    o3d.visualization.draw_geometries(models)
