import open3d as o3d
import numpy as np
import pcl

# 输入点云数据，可视化点云
def visualize_point_cloud(pcd, title="Open3D"):
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd))
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(o3d_pcd)
    vis.run()
    vis.destroy_window()

def get_pcd_from_file(file_name):
    return pcl.load(file_name)

def grid_show(file_name="combined_mesh.obj"):
    # 读取OBJ文件
    mesh = o3d.io.read_triangle_mesh(file_name)

    # 检查网格是否成功读取
    if not mesh.is_empty():
        print("Successfully read the mesh.")
    else:
        print("Failed to read the mesh.")

    # 计算法线
    mesh.compute_vertex_normals()

    # 设置统一颜色（浅灰色）
    mesh.paint_uniform_color([0.8, 0.8, 0.8])

    # 可视化网格
    o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    # 读取点云

    # 读取网格
    grid_show("combined_mesh.obj")
