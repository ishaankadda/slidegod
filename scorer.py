import clip
import torch
from PIL import Image
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, preprocess = clip.load('ViT-B/16', device=device)

imagedir = './images/exp1'


images = []
texts = ['A presentation on ethics', 'A beautiful slide', 'A presentation on business ethics', 'A presentation on business ethics with a gathering of people']

print(f'images: {os.listdir(imagedir)}')
print(f'texts: {texts}')

for imagepath in os.listdir(imagedir):
    imagepath = os.path.join(imagedir, imagepath)
    image = preprocess(Image.open(imagepath)).unsqueeze(0).to(device)
    images.append(image)

images = torch.concat(images, dim=0)
texts = clip.tokenize(texts).to(device)

image_text_scores, _ = model(images, texts)

print(image_text_scores)

