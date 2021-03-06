import config

class State:
    head = None
    tail = None
    apple = None
    body_parts = []
    score = None
    snake = None
    monster = None
    score = 0
    action = None

    def __init__(self, App):
        self.snake = App.snake
        self.apple = (App.apple.x/config.STEP_SIZE, App.apple.y/config.STEP_SIZE)
        self.score = self.snake.score
        self.head = (self.snake.head.x/config.STEP_SIZE, self.snake.head.y/config.STEP_SIZE)
        self.action = self.snake.direction

        self.body_parts = []
        # Don't include head or tail
        for i in range(1, self.snake.length - 1):
            self.body_parts.append((self.snake.x[i]/config.STEP_SIZE, self.snake.y[i]/config.STEP_SIZE))

        n = self.snake.length - 1
        self.tail = (self.snake.x[n]/config.STEP_SIZE, self.snake.y[n]/config.STEP_SIZE)

    def isonedge(self):
        if int(self.head[0]) == 0 or int(self.head[1]) == 0 or int(self.head[0]) == (config.WIDTH_TILES - 1) or int(self.head[1]) == (config.HEIGHT_TILES - 1):
            return True
        return False

    def __eq__(self, other):
        #can't compare apple until the bug of repeating states is fixed
        #if self.apple != other.apple:
        #    return False
        #can't compare these two until the bug in the game is fixed
        #if self.score != other.score:
        #    return False

        if self.snake.length != other.snake.length:
            return False

        if self.head != other.head:
            return False

        if self.tail != other.tail:
            return False

        for i,part in enumerate(self.body_parts):
            if part != other.body_parts[i]:
                return False

        return True
