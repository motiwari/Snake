from random import randint
import pygame
import time
import apple
import snake
import config
import gameengine

class App:
    windowWidth = 800
    windowHeight = 600
    snake = 0
    apple = 0

    def __init__(self):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.gameEngine = gameengine.GameEngine()
        self.snake = snake.Snake(3)
        self.apple = apple.Apple(5,5)

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.windowWidth,self.windowHeight), pygame.HWSURFACE)

        pygame.display.set_caption('SNAKE - motiwari, rschoenh, benzhou')
        self._running = True
        self._image_surf = pygame.image.load("pygame.png").convert()
        self._apple_surf = pygame.image.load("block.png").convert()

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

    def on_loop(self):
        self.snake.update()
        self.apple.update()
        
        # does snake eat apple?
        if self.gameEngine.isCollision(self.apple, self.snake.head):
            self.apple.x = randint(2,9) * config.STEP_SIZE # DON'T SPAWN ON SNAKE
            self.apple.y = randint(2,9) * config.STEP_SIZE # DON'T SPAWN ON SNAKE
            self.snake.length = self.snake.length + 1
            self.snake.score += self.apple.value
            print "Ate apple with value ", self.apple.value
            self.apple.value = 100


        # does snake collide with itself?
        for i in range(2,self.snake.length):
            # Fix this
            dummy_head = snake.Head(self.snake.x[i], self.snake.y[i])
            if self.gameEngine.isCollision(self.snake.head, dummy_head):
                print("You lose! Collision: ")
                print("x[0] (" + str(self.snake.x[0]) + "," + str(self.snake.y[0]) + ")")
                print("x[" + str(i) + "] (" + str(self.snake.x[i]) + "," + str(self.snake.y[i]) + ")")
                print "FINAL SCORE: ", self.snake.score
                exit(0)

        pass

    def on_render(self):
        self._display_surf.fill((100,100,0))
        self.snake.draw(self._display_surf, self._image_surf)
        self.apple.draw(self._display_surf, self._apple_surf)
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while( self._running ):
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            if (keys[pygame.K_RIGHT]):
                self.snake.moveRight()

            if (keys[pygame.K_LEFT]):
                self.snake.moveLeft()

            if (keys[pygame.K_UP]):
                self.snake.moveUp()

            if (keys[pygame.K_DOWN]):
                self.snake.moveDown()

            if (keys[pygame.K_ESCAPE]):
                self._running = False

            self.on_loop()
            self.on_render()

            time.sleep (50.0 / 1000.0);

        self.on_cleanup()

if __name__ == "__main__" :
    theApp = App()
    theApp.on_execute()
