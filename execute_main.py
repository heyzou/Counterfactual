import os
from PIL import Image
import torchvision.transforms as transforms

# 指定されたディレクトリ
image_dir = '/home/data/hnakai/CelebA/img_align_celeba'
output_dir = 'results'
not_blond_images = []

# CelebAの属性ファイルを読み込み、blondではない画像のリストを作成
with open('/home/data/hnakai/CelebA/list_attr_celeba.txt', 'r') as f:
    lines = f.readlines()
    headers = lines[1].strip().split()
    blond_index = headers.index('Blond_Hair')
    
    for line in lines[2:]:
        parts = line.strip().split()
        image_name = parts[0]
        if parts[blond_index] == '-1':
            not_blond_images.append(image_name)

# 指定された範囲の画像をフィルタリング
not_blond_images = [img for img in not_blond_images if '000145.jpg' <= img <= '000146.jpg']

# 出力ディレクトリが存在しない場合は作成
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 画像をリサイズする関数
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    return transform(image)

# 画像を処理
for image_name in not_blond_images:
    image_path = os.path.join(image_dir, image_name)
    if os.path.exists(image_path):
        # 画像をリサイズ
        resized_image = preprocess_image(image_path)
        
        # リサイズ後の画像を一時ファイルとして保存
        resized_image_path = os.path.join(output_dir, f"resized_{image_name}")
        resized_image_pil = transforms.ToPILImage()(resized_image)
        resized_image_pil.save(resized_image_path)

        command = f"python3 main.py main data-set --name CelebA classifier --path checkpoints/classifiers/CelebA_CNN_9.pth generative-model --g_type Flow adv-attack --image_path {resized_image_path} --target_class 1 --lr 5e-3 --num_steps 1000 --save_at 0.99"
        os.system(command)

print("Processing complete.")
