import config

class Apple:
    x = 0
    y = 0

    def __init__(self,x,y):
        self.x = x * config.STEP_SIZE
        self.y = y * config.STEP_SIZE

    def draw(self, surface, image):
        surface.blit(image, (self.x, self.y))
