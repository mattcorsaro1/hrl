'''
Copyright, 2021, Matt Corsaro, matthew_corsaro@brown.edu
'''

import math
import numpy as np

from PIL import Image as PIL_Image

import open3d as o3d

from utils.mj_transforms import *

"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height

@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels

@return t_mat:  4x4 transformation matrix as numpy array
"""
def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

"""
For visualization purposes, creates an o3d mesh at a specified pose and scales
    down axes size.

@param pose:         4x4 transformation matrix
@param scale_down:   number of times smaller axes should appear

@return scaled_axes: o3d TriangleMesh with scaled axes at specified pose
"""
def o3dTFAtPose(pose, scale_down=10):
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    scaling_maxtrix = np.ones((4,4))
    scaling_maxtrix[:3, :3] = scaling_maxtrix[:3, :3]/scale_down
    scaled_pose = pose*scaling_maxtrix
    axes.transform(scaled_pose)
    return axes

"""
Class that renders depth images in MuJoCo, processes depth images from
    multiple cameras, converts them to point clouds, and processes the point
    clouds
"""
class PointCloudGenerator(object):
    """
    initialization function

    @param sim:       MuJoCo simulation object
    @param min_bound: If not None, list len(3) containing smallest x, y, and z
        values that will not be cropped
    @param max_bound: If not None, list len(3) containing largest x, y, and z
        values that will not be cropped
    """
    def __init__(self, sim, min_bound=None, max_bound=None):
        super(PointCloudGenerator, self).__init__()

        self.sim = sim

        # I think these can be set to anything
        self.img_width = 640
        self.img_height = 480

        self.cam_names = self.sim.model.camera_names

        self.target_bounds=None
        if min_bound != None and max_bound != None:

            # hack for version 0.8.0.0 on CCV
            if int(o3d.__version__.split('.')[1]) == 8: 
                self.target_bounds = o3d.geometry.AxisAlignedBoundingBox()
                self.target_bounds.max_bound = max_bound
                self.target_bounds.min_bound = min_bound
                self.min_bound = min_bound
                self.max_bound = max_bound
            else:
                self.target_bounds = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

        # List of camera intrinsic matrices
        self.cam_mats = []
        for cam_id in range(len(self.cam_names)):
            fovy = math.radians(self.sim.model.cam_fovy[cam_id])
            f = self.img_height / (2 * math.tan(fovy / 2))
            cam_mat = np.array(((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1)))
            self.cam_mats.append(cam_mat)

    def generateCroppedPointCloud(self, save_img_dir=None):
        o3d_clouds = []
        cam_poses = []
        for cam_i in range(len(self.cam_names)):
            # Render and optionally save image from camera corresponding to cam_i
            depth_img = self.captureImage(cam_i)
            # If directory was provided, save color and depth images
            #    (overwriting previous)
            if save_img_dir != None:
                self.saveImg(depth_img, save_img_dir, "depth_test_" + str(cam_i))
                color_img = self.captureImage(cam_i, False)
                self.saveImg(color_img, save_img_dir, "color_test_" + str(cam_i))

            # convert camera matrix and depth image to Open3D format, then
            #    generate point cloud
            od_cammat = cammat2o3d(self.cam_mats[cam_i], self.img_width, self.img_height)
            od_depth = o3d.geometry.Image(depth_img)
            o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)

            # Compute world to camera transformation matrix
            cam_body_id = self.sim.model.cam_bodyid[cam_i]
            cam_pos = self.sim.model.body_pos[cam_body_id]
            c2b_r = rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_i])
            # In MuJoCo, we assume that a camera is specified in XML as a body
            #    with pose p, and that that body has a camera sub-element
            #    with pos and euler 0.
            #    Therefore, camera frame with body euler 0 must be rotated about
            #    x-axis by 180 degrees to align it with the world frame.
            b2w_r = quat2Mat([0, 1, 0, 0])
            c2w_r = np.matmul(c2b_r, b2w_r)
            c2w = posRotMat2Mat(cam_pos, c2w_r)
            # TODO(mcorsaro): I think this modifies the original cloud.. is that ok?
            transformed_cloud = o3d_cloud.transform(c2w)

            # If both minimum and maximum bounds are provided, crop cloud to fit
            #    inside them.
            if self.target_bounds != None:
                
                # hack for using open3d 0.8.0 on CCV
                if int(o3d.__version__.split('.')[1]) == 8: 
                    transformed_cloud = transformed_cloud.crop(self.min_bound, self.max_bound)
                else: 
                    transformed_cloud = transformed_cloud.crop(self.target_bounds)

            # Estimate normals of cropped cloud, then flip them based on camera
            #    position.
            transformed_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=250))
            num_points, num_normals = np.asarray(transformed_cloud.points).shape[0], np.asarray(transformed_cloud.normals).shape[0]
            if num_points > 0 and num_points == num_normals:
                transformed_cloud.orient_normals_towards_camera_location(cam_pos)
                o3d_clouds.append(transformed_cloud)

        combined_cloud = o3d.geometry.PointCloud()
        for cloud in o3d_clouds:
            combined_cloud += cloud

        return combined_cloud

    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def depthimg2Meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def verticalFlip(self, img):
        return np.flip(img, axis=0)

    # Render and process an image
    def captureImage(self, cam_ind, capture_depth=True):
        rendered_images = self.sim.render(self.img_width, self.img_height, camera_name=self.cam_names[cam_ind], depth=capture_depth)
        if capture_depth:
            img, depth = rendered_images
            depth = self.verticalFlip(depth)
            real_depth = self.depthimg2Meters(depth)

            return real_depth
        else:
            img = rendered_images
            # Rendered images appear to be flipped about vertical axis
            return self.verticalFlip(img)

    # Normalizes an image so the maximum pixel value is 255,
    # then writes to file
    def saveImg(self, img, filepath, filename):
        normalized_image = img/img.max()*255
        normalized_image = normalized_image.astype(np.uint8)
        im = PIL_Image.fromarray(normalized_image)
        im.save(filepath + '/' + filename + ".jpg")
