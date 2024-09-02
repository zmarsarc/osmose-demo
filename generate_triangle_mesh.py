import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

# 进行旋转，使平面与y轴垂直
def rotate_point_cloud_to_align_plane_with_y_axis(plane_coefficients, pcd_file):
    # 提取平面法向量
    plane_normal = np.array(plane_coefficients[:3])
    plane_normal = plane_normal / np.linalg.norm(plane_normal)  # 归一化

    # y 轴方向的单位向量
    y_axis = np.array([0, 1, 0])

    # 计算旋转轴（法向量与 y 轴的叉积）
    rotation_axis = np.cross(plane_normal, y_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # 归一化

    # 计算旋转角度（法向量与 y 轴的夹角）
    cos_angle = np.dot(plane_normal, y_axis)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # 创建旋转矩阵
    rotation_vector = rotation_axis * angle
    rotation = R.from_rotvec(rotation_vector)
    rotation_matrix = rotation.as_matrix()

     # 读取点云数据
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)

    # 对点云数据进行旋转
    rotated_points = np.dot(points, rotation_matrix.T)

    pcd_cylinder_points = np.dot(np.asarray(o3d.io.read_point_cloud("cylinder.pcd").points), rotation_matrix.T)
    # 检查是否半数的点都满足 y 值大于等于零，否则需要对所有点云y值取负值
    y_values = pcd_cylinder_points[:, 1]
    if np.sum(y_values >= 0) < len(y_values) / 2:
        rotated_points[:, 1] = -rotated_points[:, 1]
        print("进行上下翻转")

    # 创建新的点云对象
    rotated_pcd = o3d.geometry.PointCloud()
    rotated_pcd.points = o3d.utility.Vector3dVector(rotated_points)

    # 保存并展示旋转后的点云数据
    o3d.io.write_point_cloud("output/rotated_combined.pcd", rotated_pcd)

    return rotated_pcd

def merge_and_convert_to_mesh(plane_coefficients, plane_pcd_path, cylinder_pcd_path, output_mesh_path="output/combined_mesh.obj"): 
    # 加载平面点云
    plane_pcd = o3d.io.read_point_cloud(plane_pcd_path)
    # 加载圆柱点云
    cylinder_pcd = o3d.io.read_point_cloud(cylinder_pcd_path)
       
    # 合并点云
    combined_pcd = plane_pcd + cylinder_pcd
    
    # 保存合并后的点云为 .pcd 文件
    o3d.io.write_point_cloud("combined.pcd", combined_pcd)
    
    rotated_combined_pcd = rotate_point_cloud_to_align_plane_with_y_axis(plane_coefficients, "combined.pcd")
        
    # 计算法线
    rotated_combined_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # 将点云转换为三角网格
    distances = rotated_combined_pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        rotated_combined_pcd,
        o3d.utility.DoubleVector([radius, radius * 2])
    )
    
    # 保存三角网格为 .obj 文件
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print("Saved mesh to", output_mesh_path)
    

if __name__ == "__main__":
    plane_pcd_path = "filtered_plane.pcd"
    cylinder_pcd_path = "cylinder.pcd"
    output_mesh_path = "combined_mesh.obj"  # 将文件扩展名改为 .obj
    
    merge_and_convert_to_mesh(plane_pcd_path, cylinder_pcd_path, output_mesh_path)
