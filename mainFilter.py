import pcl
import numpy as np
from configparser import ConfigParser
import json
import os
# 下面是一些自己的工具函数
from utils import visualize_point_cloud,get_pcd_from_file
from target_height import extract_cylinder_section_at_height, get_radius_list
from elliptic import fit_and_plot_ellipse
from generate_triangle_mesh import merge_and_convert_to_mesh

# 距离过滤，初步筛选点云，否则别说拟合速度很慢，失败率也很高
def filter_point_cloud(pcd, min_distance=0.6, max_distance=1.5):
    distances = np.linalg.norm(np.asarray(pcd), axis=1)
    indices = np.where((distances > min_distance) & (distances <= max_distance))[0]
    filtered_cloud = pcl.PointCloud()
    filtered_cloud.from_array(np.asarray(pcd)[indices])
    return filtered_cloud

# 地面拟合，RANSAC算法迭代的结果
def RANSAC_FIT_Plane(pcd, distance_threshold=0.01):
    seg = pcd.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(1000)
    seg.set_distance_threshold(distance_threshold)

    # Segment the plane
    indices, coefficients = seg.segment()

    if len(indices) == 0:
        print("Could not estimate a planar model for the given dataset.")
        return None, None

    plane_pcd = pcd.extract(indices, negative=False)
    
    # 打印平面的法线
    normal = coefficients[:3]
    pcl.save(plane_pcd, "plane.pcd")
    print(f"Plane normal: {normal}")
    print("Plane coefficients:", coefficients)

    return plane_pcd, coefficients

# 圆柱体拟合，同样选择RANSAC算法
def RANSAC_FIT_Cylinder(pcd, distance_threshold=0.025, radius_min=0.05, radius_max=0.2):
    seg = pcd.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_CYLINDER)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(1000)
    seg.set_distance_threshold(distance_threshold)
    seg.set_radius_limits(radius_min, radius_max)

    # Segment the cylinder
    indices, coefficients = seg.segment()

    if len(indices) == 0:
        print("Could not estimate a cylindrical model for the given dataset.")
        return None, None

    cylinder_pcd = pcd.extract(indices, negative=False)
    
    pcl.save(cylinder_pcd, "cylinder.pcd")
    
    # 打印圆柱体的半径和法线
    radius = coefficients[6]
    axis = coefficients[3:6]
    print(f"Cylinder radius: {radius} meters")
    print(f"Cylinder axis: {axis} (unit vector)")
    print("Cylinder coefficients:", coefficients)
    
    return cylinder_pcd, coefficients

# 计算地面法向量和圆柱体轴线之间的夹角
def calculate_angle_between_vectors(plane_coefficients, cylinder_coefficients):
    # 计算地面法线和圆柱体轴线的夹角
    v1 = plane_coefficients[:3]
    v2 = cylinder_coefficients[3:6]
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = dot_product / norms
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    # 确保角度为锐角
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    return angle_deg

# 新增需求，目的是重新选定地面点云以圆柱体的轴线与地面平面交点为球心再重新做地面拟合
def extract_points_near_plane(pcd, plane_coefficients, cylinder_coefficients, radius=1.0):
    # 提取平面法向量和圆柱轴线
    plane_normal = np.array(plane_coefficients[:3])
    plane_d = plane_coefficients[3]
    cylinder_point = np.array(cylinder_coefficients[:3])
    cylinder_axis = np.array(cylinder_coefficients[3:6])

    # 计算圆柱轴线与地面平面的交点
    t = -(np.dot(plane_normal, cylinder_point) + plane_d) / np.dot(plane_normal, cylinder_axis)
    intersection_point = cylinder_point + t * cylinder_axis
    
    # 计算交点到(0,0,0)的平移向量
    translation_vector = -intersection_point

    # 计算每个点到交点的距离
    points = np.asarray(pcd, dtype=np.float32)  # 确保数据类型为 float32
    distances = np.linalg.norm(points - intersection_point, axis=1)

    # 筛选出距离小于指定半径的点
    indices = np.where(distances <= radius)[0]
    filtered_points = points[indices]

    # 创建新的点云对象
    filtered_pcd = pcl.PointCloud()
    filtered_pcd.from_array(filtered_points)

    # 对筛选后的点云进行地面拟合
    plane_pcd, new_plane_coefficients = RANSAC_FIT_Plane(filtered_pcd)
    
    # 对拟合后的点云做平移操作
    if plane_pcd is not None:
        plane_points = np.asarray(plane_pcd, dtype=np.float32)  # 确保数据类型为 float32
        translated_plane_points = plane_points + translation_vector.astype(np.float32)  # 确保数据类型为 float32
        translated_plane_pcd = pcl.PointCloud()
        translated_plane_pcd.from_array(translated_plane_points)
        pcl.save(translated_plane_pcd, "filtered_plane.pcd")
        
        # 读取圆柱体点云做平移操作
        cylinder_pcd = pcl.load("cylinder.pcd")
        cylinder_points = np.asarray(cylinder_pcd, dtype=np.float32)  # 确保数据类型为 float32
        translated_cylinder_points = cylinder_points + translation_vector.astype(np.float32)  # 确保数据类型为 float32
        translated_cylinder_pcd = pcl.PointCloud()
        translated_cylinder_pcd.from_array(translated_cylinder_points)
        pcl.save(translated_cylinder_pcd, "cylinder.pcd")

# 执行主函数
if __name__ == "__main__":
    # 创建配置解析器
    config = ConfigParser()
    # 读取配置文件
    config.read('config.ini')
    # 此处修改读取的点云文件名
    pcd = get_pcd_from_file("result.pcd")
    # 读取点云并展示
    filtered_pcd = filter_point_cloud(pcd, config.getfloat("Settings", "min_distance"), config.getfloat("Settings", "max_distance"))
    
    # 拟合平面并打印法线
    plane_pcd, plane_coefficients = RANSAC_FIT_Plane(filtered_pcd, config.getfloat("Settings", "distance_threshold_plane"))
    
    # 拟合圆柱体并打印半径和法线
    cylinder_pcd, cylinder_coefficients = RANSAC_FIT_Cylinder(filtered_pcd, config.getfloat("Settings", "distance_threshold_cylinder"), config.getfloat("Settings", "radius_min"), config.getfloat("Settings", "radius_max"))
    if cylinder_pcd is not None:

        angle = calculate_angle_between_vectors(plane_coefficients, cylinder_coefficients)

        print(f"Angle between plane normal and cylinder axis: {angle:.2f} degrees")
        
        # 提取平面附近的点云数据并保存
        extract_points_near_plane(pcd, plane_coefficients, cylinder_coefficients, config.getfloat("Ball", "radius"))
        
        # 获取某一高度的截面图像
        # extract_cylinder_section_at_height(cylinder_pcd, plane_coefficients, config.getfloat("section", "height"))
        # 获取各个高度的椭圆短半径并输出高度
        radius_list, height = get_radius_list(cylinder_pcd, plane_coefficients)
        print("Height:", height)
        # 点云数据转换为三角网格
        merge_and_convert_to_mesh(plane_coefficients, "filtered_plane.pcd", "cylinder.pcd")
        
        # 创建一个字典
        data = {}
        data.update({"id": 0, 
                     "location": "unknow location", 
                     "description": "no description", 
                     "mesh": "combined_mesh.obj"})
        data.update({"leaning": angle})
        data.update({"radius": cylinder_coefficients[6]})
        data.update({"height": height})
        data.update({"diameter": radius_list})
        
        # 确保 output 目录存在
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        # 保存数据到json文件
        with open('output/pole-0.json', 'w') as f:
            json.dump(data, f, indent=4)


        
        
        
        
