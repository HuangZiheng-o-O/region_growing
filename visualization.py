# visualization.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_results(original_image, segmented_image):
    """
    显示原始图像和分割结果。

    参数：
        original_image: 原始图像（numpy数组）。
        segmented_image: 分割后的二值图像（numpy数组）。
    """
    # 创建掩膜，将分割结果叠加到原始图像上
    mask = np.zeros_like(original_image)
    mask[segmented_image == 1] = [0, 0, 255]  # 使用红色显示分割区域

    # 将原始图像和掩膜叠加
    overlay = cv2.addWeighted(original_image, 0.7, mask, 0.3, 0)

    # 使用 Matplotlib 显示结果
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(segmented_image, cmap='gray')
    plt.title('分割结果')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('分割区域叠加')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
