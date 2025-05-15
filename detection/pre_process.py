import os

# 1. 源视频文件夹与目标输出文件夹
folder = '/data0/user/ycliu/AISA_data/Celeb-DF-v2/YouTube-real'
  # 如果想和源文件同一目录，也可直接填同一个路径


# 3. 预设的时间戳（秒）
timestamps = [0, 60, 120]

# 4. 遍历所有 .mp4 文件
for fname in os.listdir(folder):
    if not fname.lower().endswith('.mp4'):
        continue
    base = os.path.splitext(fname)[0]                # 去掉 .mp4
    output_path = "/data0/user/ycliu/AISA_data/Celeb-DF-v2/target_image_list_3.txt"
    
    # 5. 打开对应的 .txt 文件写入三行数据
    with open(output_path, 'a', encoding='utf-8') as f:
        for t in timestamps:
            # {:05d} 保证 5 位，不足前面补 0
            f.write(f"{base}_{t:05d}\n")
    
    print(f"已生成：{output_path}")
