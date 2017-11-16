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

    def __init__(self, App):
        self.snake = App.snake
        self.apple = (App.apple.x/config.STEP_SIZE, App.apple.y/config.STEP_SIZE)
        self.score = self.snake.score
        self.head = (self.snake.head.x/config.STEP_SIZE, self.snake.head.y/config.STEP_SIZE)

        self.body_parts = []
        # Don't include head or tail
        for i in range(1, self.snake.length - 1):
            self.body_parts.append((self.snake.x[i]/config.STEP_SIZE, self.snake.y[i]/config.STEP_SIZE))

        n = self.snake.length - 1
        self.tail = (self.snake.x[n]/config.STEP_SIZE, self.snake.y[n]/config.STEP_SIZE)

    def __eq__(self, other):
        if self.apple != other.apple:
            return False

        if self.score != other.score:
            return False

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
