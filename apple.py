import config

class Apple:
    x = 0
    y = 0
    image = None
    isCollidable = True
    value = config.INITIAL_APPLE_VALUE
    discountFactor = config.DISCOUNT_FACTOR

    def __init__(self, x, y):
        self.x = x * config.STEP_SIZE
        self.y = y * config.STEP_SIZE
        self.image = config.APPLE_IMAGE
        self.value = 100

    def update(self):
        self.value *= self.discountFactor

    def draw(self, surface, image):
        surface.blit(image, (self.x, self.y))
