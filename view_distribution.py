import open3d as o3d
import numpy as np
from utils import load_view_point, save_view_point
print("Testing IO for point cloud ...")
points = np.fromfile("data/bin/000214.bin", dtype=np.float32).reshape(-1, 4)
pts = o3d.geometry.PointCloud()
pts.points = o3d.utility.Vector3dVector(points[:,:3])
# pts.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(points.shape[0], 3)))
# pts.estimate_normals()

voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pts, voxel_size=1.0)
voxels_all= voxels.get_voxels()
# print(voxels_all)
# o3d.visualization.draw_geometries([voxels])
view_thing = [pts, voxels]
load_view_point(view_thing, "data/viewpoint.json")
print("All success")