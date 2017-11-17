import config

class Head:
    x = 0
    y = 0
    isCollidable = True


    def __init__(self, x, y):
        self.x = x
        self.y = y

class Snake:
    step = config.STEP_SIZE

    isCollidable = True


    def __init__(self, length):
        self.x = [0]
        self.y = [0]
        self.ars = []
        self.direction = config.RIGHT
        # The direction a snake has can change multiple times (with the pressing
        # of multiple keys) in a single timestep, which can be used to reverse
        # a snake on itself. The last_moved direction is the *actual* step it took
        # last turn.

        self.last_moved = config.RIGHT
        self.score = 0
        self.updateCountMax = 2
        self.updateCount = 0

        self.length = length
        for i in range(0,2000):
            self.x.append(-100)
            self.y.append(-100)

        self.head = Head(self.x[0], self.x[0])

    def update(self):
        self.updateCount = self.updateCount + 1
        if self.updateCount > self.updateCountMax:

            # update previous positions
            for i in range(self.length-1,0,-1):
                self.x[i] = self.x[i-1]
                self.y[i] = self.y[i-1]

            # update position of head of snake
            if self.direction == 0:
                self.x[0] = self.x[0] + self.step
            if self.direction == 1:
                self.x[0] = self.x[0] - self.step
            if self.direction == 2:
                self.y[0] = self.y[0] - self.step
            if self.direction == 3:
                self.y[0] = self.y[0] + self.step

            self.last_moved = self.direction

            self.updateCount = 0

            self.head.x = self.x[0]
            self.head.y = self.y[0]

    def moveRight(self):
        self.direction = config.RIGHT

    def moveLeft(self):
        self.direction = config.LEFT

    def moveUp(self):
        self.direction = config.UP

    def moveDown(self):
        self.direction = config.DOWN

    def draw(self, surface, image):
        for i in range(0,self.length):
            surface.blit(image,(self.x[i],self.y[i]))

    def addActionAndReward(self, action, reward):
        self.ars.append((action, reward))
