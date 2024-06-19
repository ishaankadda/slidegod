import torch
from PIL import Image
import clip

from canvas_utils import *
from gradient_utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, preprocess = clip.load('ViT-B/16', device=device)

blank_canvas = Image.new("RGB", (1920, 1080), "rgb(255, 255, 255)")

grad_estim_delta = 10
ascent_step_size = 10
max_iters = 1000

def score_image(image, prompt:str="A beautiful slide about ethics, centred text"):
    prompt = clip.tokenize([prompt]).to(device)
    image = preprocess(image).unsqueeze(0).to(device)
    score = model(image, prompt)[0][0][0].item()
    return score

elements = [
    SlideElement(image_path='../sources/ethicstext.png',
                 xy_position=[1280-640//2, 540 - 100], wh_size=[240, 240]),
    SlideElement(image_path='../sources/people.png',
                 xy_position=[0, 0], wh_size=[960, 1080]),
]

slide = Slide([255, 255, 255], elements)
slide.render().save('render.png')

for i in range(max_iters):
    estimate_slide_gradients(slide, score_image, grad_estim_delta)
    ascent_slide_gradients(slide, ascent_step_size)
    print([[ele.xy_position, ele.wh_size] for ele in slide.slideelements])
    slide.render().save('render2.png')
    print(i)










# canvas.save('temp.png')