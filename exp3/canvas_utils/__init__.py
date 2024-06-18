import PIL
from PIL import Image
from typing import List, Union, Annotated


class SlideElement:
    def __init__(self, image_path:str, xy_position:Annotated[List[int], 2], wh_size:Annotated[List[int], 2]):
        self.image_path = image_path
        self.xy_position = xy_position
        self.wh_size = wh_size
    
    def drawselfoncanvas(self, canvas:PIL.Image.Image) -> PIL.Image.Image:
        """Overlay the image onto canvas at the appropriate scaling and coordinates."""
        image1 = Image.open(self.image_path).resize(self.wh_size, Image.ANTIALIAS)
        overlaidcanvas = canvas.copy
        overlaidcanvas.paste(image1, self.xy_position, image1)
        return overlaidcanvas


class Slide:
    def __init__(self, bg_color:Annotated[List[int], 3], slideelements:List[SlideElement]):
        self.bg_color = bg_color
        self.canvas = Image.new("RGB", (1920, 1080), f"rgb{tuple(self.bg_color)}")
        self.slideelements = slideelements if slideelements is not None else []
    
    def drawslide(self) -> PIL.Image.Image:
        return self.canvas.copy()
