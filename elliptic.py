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
    axes = ellipse[1]
    
    # 返回短半径，需要的话可取用
    # 由于需要直径，就不/2了
    return min(axes)

if __name__ == "__main__":
    fit_and_plot_ellipse()
