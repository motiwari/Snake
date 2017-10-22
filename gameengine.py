import config

class GameEngine:
    def isCollision(self, obj1, obj2):
        if obj1.isCollidable and obj2.isCollidable:
            if obj1.x >= obj2.x and obj1.x <= obj2.x + config.STEP_SIZE - 1:
                if obj1.y >= obj2.y and obj1.y <= obj2.y + config.STEP_SIZE - 1:
                    return True
        return False
