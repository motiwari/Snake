import config

class GameEngine:
    def isCollision(self, obj1, obj2):
        if obj1.isCollidable and obj2.isCollidable:
            if obj1.x >= obj2.x and obj1.x <= obj2.x + config.STEP_SIZE - 1:
                if obj1.y >= obj2.y and obj1.y <= obj2.y + config.STEP_SIZE - 1:
                    return True
        return False

    def getBoardFreeSquares(self, snake):
        free_squares = []
        for i in range(0, config.WIDTH_TILES):
            for j in range(0, config.HEIGHT_TILES):
                if not (config.STEP_SIZE*i in snake.x and config.STEP_SIZE*j in snake.y):
                    free_squares.append((config.STEP_SIZE*i,config.STEP_SIZE*j))
        return free_squares
