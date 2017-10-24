from random import randint
import pygame
import time
import apple
import snake
import config
import gameengine
import sys

class App:
    windowWidth = config.DEFAULT_WINDOW_WIDTH
    windowHeight = config.DEFAULT_WINDOW_HEIGHT
    boardWidth = windowWidth/config.STEP_SIZE
    boardHeight = windowHeight/config.STEP_SIZE
    snake = 0
    apple = 0

    def __init__(self,usingAI = False):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.gameEngine = gameengine.GameEngine()
        self.snake = snake.Snake(3)
        self.apple = apple.Apple(5,5)
        self.usingAI = usingAI

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
        # does snake collide with itself?
        for i in range(2,self.snake.length):
            # Fix this
            dummy_head = snake.Head(self.snake.x[i], self.snake.y[i])
            if self.gameEngine.isCollision(self.snake.head, dummy_head):
                self.snake.addActionAndReward(self.snake.direction, 0)
                print("You lose! Collision: ")
                print "FINAL SCORE: ", self.snake.score
                print self.snake.ars
                exit(0)

        if self.snake.head.x < 0 or self.snake.head.x >= self.windowWidth or \
            self.snake.head.y < 0 or self.snake.head.y >= self.windowHeight:
            self.snake.addActionAndReward(self.snake.direction, 0)
            print("You lose! Off the board!")
            print "FINAL SCORE: ", self.snake.score
            print self.snake.ars
            exit(0)

        # does snake eat apple?
        if self.gameEngine.isCollision(self.apple, self.snake.head):
            self.apple.x = randint(2,9) * config.STEP_SIZE # DON'T SPAWN ON SNAKE
            self.apple.y = randint(2,9) * config.STEP_SIZE # DON'T SPAWN ON SNAKE
            self.snake.length = self.snake.length + 1
            self.snake.score += self.apple.value
            print "Ate apple with value ", self.apple.value
            self.snake.addActionAndReward(self.snake.direction, self.apple.value)
            self.apple.value = 100
        else:
            self.snake.addActionAndReward(self.snake.direction, 0)




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

            if(self.usingAI):
                x = self.snake.x[0]
                y = self.snake.y[0]
                d = self.snake.direction

                if self.apple.x-x < 0 and self.snake.direction!=0:
                    print 'd', self.snake.direction
                    print(self.apple.x-x)
                    print 1
                    self.snake.moveLeft()
                elif self.apple.x-x > 0 and self.snake.direction!=1:
                    print 'd', self.snake.direction
                    print self.apple.x-x
                    print 2
                    self.snake.moveRight()
                elif self.apple.y-y < 0 and self.snake.direction!=3:
                    print 'd', self.snake.direction
                    print self.apple.y-y
                    print 3
                    self.snake.moveUp()
                elif self.apple.y-y > 0 and self.snake.direction!=2:
                    print 'd', self.snake.direction
                    print self.apple.y-y
                    print 4
                    self.snake.moveDown()
                else:
                    if self.snake.direction == 1 or self.snake.direction==0:
                        print 5
                        self.snake.direction += 2
                    else:
                        print 6
                        self.snake.direction -= 2
            else:

                if keys[pygame.K_RIGHT] and self.snake.direction != 1:
                    self.snake.moveRight()

                if keys[pygame.K_LEFT] and self.snake.direction != 0:
                    self.snake.moveLeft()

                if keys[pygame.K_UP] and self.snake.direction != 3:
                    self.snake.moveUp()

                if keys[pygame.K_DOWN] and self.snake.direction != 2:
                    self.snake.moveDown()

                if (keys[pygame.K_ESCAPE]):
                    self._running = False

            self.on_loop()
            self.on_render()

            time.sleep (50.0 / 1000.0);

        self.on_cleanup()

if __name__ == "__main__" :
    theApp = App(sys.argv[1] == 'ai')
    theApp.on_execute()
