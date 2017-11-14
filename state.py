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
        for i in range(self.snake.length - 1):
            self.body_parts.append((self.snake.x[i]/config.STEP_SIZE, self.snake.y[i]/config.STEP_SIZE))

        n = self.snake.length - 1
        self.tail = (self.snake.x[n]/config.STEP_SIZE, self.snake.y[n]/config.STEP_SIZE)
