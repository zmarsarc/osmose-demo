import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def fit_and_plot_ellipse(points, plot_title="Scatter Plot of Transformed Points with Fitted Ellipse"):
    # 解析数据
    points = np.array(points, dtype=np.float32)
    
    # 检查点的数量
    if len(points) < 10:
        print("点的数量少于10个,拟合失败")
        return 0

    # 使用 OpenCV 进行椭圆拟合
    ellipse = cv2.fitEllipse(points)

    # 获取椭圆参数
    # center = ellipse[0]
    axes = ellipse[1]
    # angle = ellipse[2]

    # 打印长短半径的长度
    # print(f"长半径: {max(axes)/2}")
    # print(f"短半径: {min(axes)/2}")

    # 创建绘图
    # fig, ax = plt.subplots()

    # 绘制原始点
    # ax.scatter(points[:, 0], points[:, 1], color="magenta", marker="+", s=80, label="samples")

    # 绘制拟合椭圆
    # ellipse_patch = Ellipse(xy=center, width=axes[0], height=axes[1], angle=angle, edgecolor='deepskyblue', fc='None', lw=2, label='fitted ellipse')
    # ax.add_patch(ellipse_patch)

    # 设置标签和标题
    # plt.xlabel("$x$")
    # plt.ylabel("$y$")
    # plt.legend()
    # plt.title(plot_title)
    # plt.axis('equal')
    # plt.show()
    
    # 返回短半径，需要的话可取用
    return min(axes)/2

if __name__ == "__main__":
    fit_and_plot_ellipse()
