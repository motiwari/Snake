import config

class Apple:
    x = 0
    y = 0
    image = None
    isCollidable = True

    def __init__(self,x,y):
        self.x = x * config.STEP_SIZE
        self.y = y * config.STEP_SIZE
        self.image = config.APPLE_IMAGE
        
    def draw(self, surface, image):
        surface.blit(image, (self.x, self.y))
