import PIL
from PIL import Image
from typing import List, Union, Annotated, Callable
from canvas_utils import Slide, SlideElement
from random import random
from math import ceil


# def estimate_slide_gradients(slide:Slide, score_fn, delta:int):
#     """estimate the gradients for all the slide elements given all the paramaters given a scoring function."""
#     s0 = score_fn(slide.render())
#     for ele in slide.slideelements:
#         for delta_val in [ceil(delta*i/5) for i in range(1, 5)]:
#             ele.xy_grads[0] *= 0
#             ele.wh_grads[0] *= 0
#             ele.xy_grads[1] *= 0
#             ele.wh_grads[1] *= 0
#             for i in range(len(ele.xy_position)):
#                 ele.xy_position[i] += delta_val
#                 s_ele_i = score_fn(slide.render())
#                 ele.xy_position[i] -= delta_val
#                 ele.xy_grads[i] += (s_ele_i - s0) / delta_val
#                 print((s_ele_i - s0) / delta_val)
#             for i in range(len(ele.wh_size)):
#                 ele.wh_size[i] += delta_val
#                 s_ele_i = score_fn(slide.render())
#                 ele.wh_size[i] -= delta_val
#                 ele.wh_grads[i] += (s_ele_i - s0) / delta_val
#                 print((s_ele_i - s0) / delta_val)

# def ascent_slide_gradients(slide:Slide, step_size:int):
#     for ele in slide.slideelements:
#         for i in range(len(ele.xy_position)):
#             gradval = ele.xy_grads[i]
#             update_yes = int(abs(gradval) > random())
#             if ele.xy_grads[i] is not None:
#                 print('ascent!', ele.xy_grads[i])
#                 ele.xy_position[i] += (step_size if ele.xy_grads[i] > 0 else -step_size)
#         for i in range(len(ele.wh_grads)):
#             if ele.wh_grads[i] is not None:
#                 print('ascent!', ele.wh_grads[i])
#                 ele.wh_size[i] += (step_size if ele.wh_grads[i] > 0 else -step_size)
#     print()

def estimate_grad_ascent(slide:Slide, step_size:int, score_fn_batched, delta:int):
    """estimate and step the gradients for all the slide elements given all the paramaters given a scoring function."""
    s0 = score_fn_batched([slide.render()])[0]
    for ele in slide.slideelements:
        for delta_val in [ceil(delta*i/2) for i in range(1, 5)]:
            for i in range(len(ele.xy_position)):
                ele.xy_position[i] += delta_val
                s_ele_i = score_fn_batched([slide.render()])[0]
                gradval = (s_ele_i - s0) / delta_val
                if random()/6 < abs(gradval):
                    ele.xy_position[i] += step_size if gradval > 0 else -step_size
                ele.xy_position[i] -= delta_val
                # print((s_ele_i - s0) / delta_val)

            for i in range(len(ele.wh_size)):
                ele.wh_size[i] += delta_val
                s_ele_i = score_fn_batched([slide.render()])[0]
                gradval = (s_ele_i - s0) / delta_val
                if random()/6 < abs(gradval):
                    ele.wh_size[i] += step_size if gradval > 0 else -step_size
                ele.wh_size[i] -= delta_val
                # print((s_ele_i - s0) / delta_val)
                

    
