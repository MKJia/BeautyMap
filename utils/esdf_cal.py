#
# Copyright 2022 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
from scipy.spatial import cKDTree as KDTree
import open3d as o3d
from .voxel_grid import VoxelGrid

def generate_esdf_from_pts(gt_pts: o3d.geometry.PointCloud, points_xyz: np.ndarray) -> VoxelGrid:
    """Generates an ESDF from a triangle mesh at voxel centers passed in. This is
       performed by looking up the closest vertices on the input mesh.

    Args:
        gt_pts (o3d.geometry.TriangleMesh): Mesh defining the surface.
        points_xyz (np.ndarray): Voxel centers where we want the ESDF calculated.

    Returns:
        VoxelGrid: A voxel grid containing the groundtruth ESDF values.
    """    
    # Getting the GT distances using a KD-tree to find the closest points on the mesh surface.
    gt_kdtree = KDTree(gt_pts.points)
    gt_distances, gt_indices = gt_kdtree.query(points_xyz)
    # Getting the signs of the distances
    gt_closest_points = np.asarray(gt_pts.points)[gt_indices]
    gt_closest_vectors = points_xyz - gt_closest_points
    gt_closest_normals = np.asarray(gt_pts.normals)[gt_indices]
    dots = np.sum(np.multiply(gt_closest_vectors, gt_closest_normals), axis=1)
    signs = np.where(dots >= 0, 1.0, -1.0)
    gt_distances = np.multiply(gt_distances, signs)
    return VoxelGrid.createFromSparseVoxels(points_xyz, gt_distances)