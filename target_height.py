import matplotlib.pyplot as plt
import numpy as np
from elliptic import fit_and_plot_ellipse

# 截取相对于地面的某一高度的圆柱体的点云截面,需要圆柱体的点云数据，地面平面系数，目标高度和宽容度(因为就某一个特定高度的点很少，取一个小范围)
def extract_cylinder_section_at_height(cylinder_pcd, ground_plane_coefficients, target_height,Tolerance = 0.004):
    def get_section_points(cylinder_pcd, ground_plane_coefficients, target_height):
        # 提取地面平面系数
        a, b, c, d = ground_plane_coefficients

        # 计算目标高度范围
        lower_bound = target_height - Tolerance
        upper_bound = target_height + Tolerance

        # 提取目标高度范围内的点云
        points = np.asarray(cylinder_pcd)
        distances = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
        section_indices = np.where((distances >= lower_bound) & (distances <= upper_bound))[0]
        section_points = points[section_indices]

        return section_points

    # 获取目标高度的截面点云
    section_points = get_section_points(cylinder_pcd, ground_plane_coefficients, target_height)

    # 如果点数不超过10个，取-target_height重新执行一遍
    if len(section_points) <= 10:
        section_points = get_section_points(cylinder_pcd, ground_plane_coefficients, -target_height)
        if len(section_points) <= 10:
            # print("未识别此高度下的圆柱数据")
            return 0, None

    # 重塑坐标系
    # 地面法向量作为z轴
    a, b, c, d = ground_plane_coefficients
    z_axis = np.array([a, b, c])
    z_axis = z_axis / np.linalg.norm(z_axis)

    # 随意选择一个向量作为新的x轴
    x_axis = np.array([1, 0, -a/c]) if c != 0 else np.array([0, 1, -b/c])
    x_axis = x_axis / np.linalg.norm(x_axis)

    # 计算新的y轴
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # 变换矩阵
    transformation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

    # 将点云数据变换到新的坐标系
    transformed_points = (section_points - np.array([0, 0, -d/c])).dot(transformation_matrix)
    
    # 保留x和y轴信息
    transformed_points = transformed_points[:, :2]
        
    return fit_and_plot_ellipse(transformed_points), transformed_points

# 对高度为0.01开始递增每次0.01m，一直执行除非连续50次截取结果返回都是0，则停止并返回计算的半径的列表，当然去除最后50次结果，同时返回最终高度-0.5
def get_radius_list(cylinder_pcd, ground_plane_coefficients):
    i = 0
    height = 0.00
    radius_list = []
    while True:
        height += 0.01
        radius,_ = extract_cylinder_section_at_height(cylinder_pcd, ground_plane_coefficients, height)
        radius_list.append(radius)
        if radius == 0:
            i += 1
        else:
            i = 0
        if i == 50:
            return radius_list[:-50], height - 0.5
