from PIL import Image, ImageDraw
import os
import random

img_dir = 'D:/FootballAI/dataset_v2/images/train'
lbl_dir = 'D:/FootballAI/dataset_v2/labels/train'

files = os.listdir(img_dir)
name = random.choice(files)
img = Image.open(f'{img_dir}/{name}')
w, h = img.size
draw = ImageDraw.Draw(img)
colors = {0: 'blue', 1: 'red', 2: 'yellow', 3: 'cyan'}

with open(f'{lbl_dir}/{name.replace(".jpg", ".txt")}') as f:
    for line in f:
        cls, xc, yc, bw, bh = map(float, line.split())
        cls = int(cls)
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        draw.rectangle([x1, y1, x2, y2], outline=colors[cls], width=3)

img.save('D:/FootballAI/label_check.jpg')
print('Saved check image:', name)