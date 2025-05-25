import numpy as np
from PIL import Image, ImageDraw

# 创建一个模拟的缺陷图像
img = Image.new('RGB', (400, 400), color = 'gray')
draw = ImageDraw.Draw(img)

# 绘制一些模拟的缺陷
draw.rectangle(((100, 100), (150, 300)), fill='black')
draw.ellipse(((200, 200), (300, 300)), fill='black')
draw.line(((50, 50), (350, 350)), fill='black', width=5)

# 保存图像
img.save('test_images/synthetic_defect.jpg')
print('创建了测试图像: test_images/synthetic_defect.jpg') 