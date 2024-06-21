# slidegod
The only AI enabled presentation maker you will ever need.

# exp1
Can CLIP score stock photo variants in a slide with known intent?


# exp2
Can CLIP score variation of element positining in an image?

# exp3
Can we automate element positioning using CLIP scoring? \
Experiment inside ./exp3
To run this experiment and replicate generation of `render.png` (initial slide) -> `render2.png` (optimised slide):
1. Navigate to `./exp3/`
2. `python runner.py`

This script tries to optimise placement of two images on a blank white slide, the first being a [stock photo of a gathering of people](./sources/people.png) and the second, a [photo of a text box about ethics](./sources/ethicstext.png). \
On running, it shows how the elements initially look (`render.png`) and then optimizes the placement and shows a preview as well (`render2.png`). \
The placement of the slides is optimised by gradient ascent done using first principles. \
An example can be found below:
1. [example of how `render.png` can look.](./exp3/example_of_render.png)
2. [example of how `render2.png` can look after optimisation loops 100 times.](./exp3/example_of_render2.png)


# exp4 (ongoing)
Can we vectorize the placement of elements on a slide canvas? \
Current idea:
- use 2d convolution with a unit step function to parametrize spatial offsetting. This will allow gradients to flow back to [x, y, z, w].