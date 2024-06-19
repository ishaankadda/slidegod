import PIL
from PIL import Image
from typing import List, Union, Annotated


class SlideElement:
    def __init__(self, image_path:str, xy_position:Annotated[List[int], 2], wh_size:Annotated[List[int], 2]):
        self.image_path = image_path
        self.xy_position = xy_position
        self.wh_size = wh_size
        self.xy_grads = [0.0 for _ in self.xy_position]
        self.wh_grads = [0.0 for _ in self.wh_size]
    
    def drawselfoncanvas(self, canvas: Image.Image) -> Image.Image:
        """Overlay the image onto canvas at the appropriate scaling and coordinates."""
        with Image.open(self.image_path) as image1:
            image1 = image1.resize(self.wh_size, Image.ANTIALIAS)
            overlaidcanvas = canvas.copy()
            # Calculate the box where image1 should be pasted
            box = (
                self.xy_position[0],
                self.xy_position[1],
            )
            overlaidcanvas.paste(image1, box)
        return overlaidcanvas


class Slide:
    def __init__(self, bg_color:Annotated[List[int], 3], slideelements:List[SlideElement]):
        self.bg_color = bg_color
        self.canvas = Image.new("RGB", (1080, 1080), f"rgb{tuple(self.bg_color)}")
        self.slideelements = slideelements if slideelements is not None else []
    
    def render(self) -> PIL.Image.Image:
        newcanvas = self.canvas.copy()
        for ele in self.slideelements:
            newcanvas = ele.drawselfoncanvas(newcanvas)
        return newcanvas
