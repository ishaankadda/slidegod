import torch
from PIL import Image
import clip

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, preprocess = clip.load('ViT-B/16', device=device)


blank_canvas = Image.new("RGB", (1920, 1080), "rgb(255, 255, 255)")

def score_image(image, prompt:str="A well positioned poswerpoint slide about ethics"):
    prompt = clip.tokenize([prompt]).to(device)
    image = preprocess(image).unsqueeze(0).to(device)
    score = model(image, prompt)[0][0][0].item()
    return score





# canvas.save('temp.png')