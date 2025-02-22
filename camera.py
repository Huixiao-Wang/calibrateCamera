import cv2
import numpy as np
import glob

# 棋盘格的行数和列数
chessboard_rows = 7-1  # 棋盘格的行数
chessboard_cols = 10-1  # 棋盘格的列数

# 棋盘格每个方格的实际尺寸（单位：毫米）
square_size = 25.0  # 每个方格边长

# 3D 点和 2D 点的存储列表
object_points = []  # 存储棋盘格在3D空间中的点
image_points = []   # 存储棋盘格在图像平面中的点

# 准备棋盘格的3D点，假设棋盘格位于z=0的平面上
object_3d_points = np.zeros((chessboard_rows * chessboard_cols, 3), np.float32)
object_3d_points[:, :2] = np.indices((chessboard_cols, chessboard_rows)).T.reshape(-1, 2)
object_3d_points *= square_size

# 读取本地图片
image_folder = "./images/"  # 替换为您的图片文件夹路径
images = glob.glob(image_folder + "*.bmp")  # 查找所有JPG图片

if not images:
    print("未找到棋盘格图片，请检查路径和文件格式。")
    exit()

print(f"找到 {len(images)} 张图片，用于标定。")

# 遍历每张图片，查找棋盘格角点
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格的角点
    ret, corners = cv2.findChessboardCorners(gray, (chessboard_cols, chessboard_rows), None)

    if ret:
        # 如果找到棋盘格角点，优化角点的亚像素精度
        corners_subpix = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        # 存储角点
        image_points.append(corners_subpix)
        object_points.append(object_3d_points)

        # 绘制棋盘格角点并显示
        cv2.drawChessboardCorners(img, (chessboard_cols, chessboard_rows), corners_subpix, ret)
        cv2.imshow("Chessboard", img)
        cv2.waitKey(500)
    else:
        print(f"未找到棋盘格角点：{fname}")

cv2.destroyAllWindows()

# 确保至少有一些图片成功找到棋盘格角点
if not object_points or not image_points:
    print("没有足够的有效棋盘格角点来标定相机。")
    exit()

# 相机标定
print("开始标定相机...")
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, gray.shape[::-1], None, None
)

if ret:
    print("标定成功！")
    print("相机内参矩阵：\n", camera_matrix)
    print("畸变系数：\n", dist_coeffs)
else:
    print("标定失败！")

# 保存标定结果
np.savez("calibration_data.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

# 校验标定结果
print("开始验证标定结果...")
total_error = 0
for i in range(len(object_points)):
    img_points2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(image_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
    total_error += error

print(f"每张图片的平均重投影误差：{total_error / len(object_points):.4f}")
