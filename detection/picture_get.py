import cv2
import os
import pdb

# 输入文件路径
target_image_list_path = '/data0/user/ycliu/AISA_data/Celeb-DF-v2/target_image_list_3.txt'
videos_folder = '/data0/user/ycliu/AISA_data/Celeb-DF-v2/YouTube-real/'  # 视频文件夹路径
output_folder = '/data0/user/ycliu/AISA_data/Celeb-DF-v2/target_pictures/results/'  # 输出图片保存文件夹

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 读取目标帧列表
with open(target_image_list_path, 'r') as file:
    lines = file.readlines()

# 遍历每一行并处理
for line in lines:
    # 去掉行尾换行符
    line = line.strip()

    # 解析每行的内容
    video_id, frame_number = line.split('_')
    
    # 构建视频文件路径
    video_filename = f"{video_id}.mp4"
    video_path = os.path.join(videos_folder, video_filename)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        continue
    
    # 设置视频读取到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))

    # 读取指定帧
    ret, frame = cap.read()
    
    if ret:
        # 保存帧为图片
        output_image_path = os.path.join(output_folder, f"YouTube-real_{video_id}_{frame_number}.png")
        cv2.imwrite(output_image_path, frame)
        print(f"成功提取并保存帧 {frame_number} 为图片: {output_image_path}")
    else:
        print(f"无法提取帧 {frame_number} 来自视频 {video_filename}")
    
    # 释放视频文件
    cap.release()

print("所有帧提取完成！")
