import torch
from PIL import Image
import clip

from canvas_utils import *
from gradient_utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, preprocess = clip.load('ViT-B/16', device=device)

grad_estim_delta = 30
ascent_step_size = 10
max_iters = 1000

def score_image(image, prompt:str="A beautiful slide about ethics, centred text"):
    prompt = clip.tokenize([prompt]).to(device)
    image = preprocess(image).unsqueeze(0).to(device)
    score = model(image, prompt)[0][0][0].item()
    return score

def score_image_batch(images, prompt:str="A beautiful slide about ethics, centred text"):
    prompt = clip.tokenize([prompt]).to(device)
    images = torch.concat([preprocess(image).unsqueeze(0).to(device) for image in images], dim=0)
    scores = model(images, prompt)[1][0].tolist()
    return scores

elements = [
    SlideElement(image_path='../sources/people.png',
                 xy_position=[150, 150], wh_size=[350, 350]),
    SlideElement(image_path='../sources/ethicstext.png',
                 xy_position=[500, 500], wh_size=[300, 300]),
]

slide = Slide([255, 255, 255], elements)
slide.render().save('render.png')

for i in range(max_iters):
    # estimate_slide_gradients(slide, score_image, grad_estim_delta)
    # ascent_slide_gradients(slide, ascent_step_size)
    estimate_grad_ascent(slide, ascent_step_size, score_image_batch, grad_estim_delta)
    print([[ele.xy_position, ele.wh_size] for ele in slide.slideelements])
    slide.render().save('render2.png')
    print(i)










# canvas.save('temp.png')