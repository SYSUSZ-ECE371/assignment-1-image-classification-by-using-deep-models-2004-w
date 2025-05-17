import os
import shutil
from sklearn.model_selection import train_test_split

# 配置路径
data_root = r'E:\flower_dataset'
output_root = "D:\桌面\新建文件夹 (4)\dataset"
split_ratio = 0.8  # 训练集比例

# 创建输出目录
os.makedirs(f"{output_root}/train", exist_ok=True)
os.makedirs(f"{output_root}/val", exist_ok=True)

# 遍历每个类别
for class_name in os.listdir(data_root):
    class_path = os.path.join(data_root, class_name)
    if not os.path.isdir(class_path):
        continue
    
    # 获取所有图片文件
    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    train_images, val_images = train_test_split(images, test_size=1-split_ratio, random_state=42)

    # 移动训练集图片
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(output_root, "train", class_name, img)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    # 移动验证集图片
    for img in val_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(output_root, "val", class_name, img)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)



# 配置路径
train_txt = os.path.join(output_root, "train.txt")
val_txt = os.path.join(output_root, "val.txt")

# 类别字典（顺序需与 classes.txt 一致）
class_to_idx = {
    "daisy": 0,
    "dandelion": 1,
    "rose": 2,
    "sunflower": 3,
    "tulip": 4
}

# 生成 train.txt
with open(train_txt, "w") as f:
    for class_name, idx in class_to_idx.items():
        class_dir = os.path.join(output_root, "train", class_name)
        for img in os.listdir(class_dir):
            img_path = os.path.join("train", class_name, img)
            f.write(f"{img_path} {idx}")

# 生成 val.txt
with open(val_txt, "w") as f:
    for class_name, idx in class_to_idx.items():
        class_dir = os.path.join(output_root, "val", class_name)
        for img in os.listdir(class_dir):
            img_path = os.path.join("val", class_name, img)
            f.write(f"{img_path} {idx}")


classes_txt = os.path.join(output_root, "classes.txt")
with open(classes_txt, "w") as f:
    # 按定义的顺序写入类别名称（直接遍历字典的键）
    for class_name in class_to_idx.keys():
        f.write(f"{class_name}")


