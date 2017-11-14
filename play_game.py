from random import randint
import pygame
import time
import apple
import snake
import config
import gameengine
import sys
import random

def isNextMoveCollision(pyg,direction):
    dummy_head = None
    if direction == config.RIGHT:
        dummy_head = snake.Head(pyg.snake.x[0] + pyg.snake.step, pyg.snake.y[0])
    if direction == config.LEFT:
        dummy_head = snake.Head(pyg.snake.x[0] - pyg.snake.step, pyg.snake.y[0])
    if direction == config.UP:
        dummy_head = snake.Head(pyg.snake.x[0], pyg.snake.y[0] - pyg.snake.step)
    if direction == config.DOWN:
        dummy_head = snake.Head(pyg.snake.x[0], pyg.snake.y[0] + pyg.snake.step)

    #Check Board collision
    if dummy_head.x < 0 or dummy_head.x >= pyg.windowWidth or \
        dummy_head.y < 0 or dummy_head.y >= pyg.windowHeight:
        return True
    #Check Snake collision
    for i in range(1,pyg.snake.length-1): #Need to account for the fact that the snake will have moved by 1, so we don't start on 3rd segment of snake
        dummy_head2 = snake.Head(pyg.snake.x[i], pyg.snake.y[i])
        if pyg.gameEngine.isCollision(dummy_head, dummy_head2):
            return True
    return False

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
        self.apple.x, self.apple.y = random.choice(self.gameEngine.getBoardFreeSquares(self.snake))
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
            self.snake.length = self.snake.length + 1
            self.snake.score += self.apple.value
            print "Ate apple with value ", self.apple.value
            self.snake.addActionAndReward(self.snake.direction, self.apple.value)
            self.apple.value = 100
            freeSqs = self.gameEngine.getBoardFreeSquares(self.snake)
            if freeSqs == []:
                print "You WON Snake!!"
                print "FINAL SCORE: ", self.snake.score
                exit(0)
            else:
                self.apple.x, self.apple.y = random.choice(freeSqs)

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
                d = self.snake.last_moved
                #See if you can move the same direction
                if d == config.RIGHT and self.apple.x - x > 0 and not isNextMoveCollision(self, config.RIGHT):
                    self.snake.moveRight()
                elif d == config.LEFT and self.apple.x - x < 0 and not isNextMoveCollision(self, config.LEFT):
                    self.snake.moveLeft()
                elif d == config.UP and self.apple.y - y < 0 and not isNextMoveCollision(self, config.UP):
                    self.snake.moveUp()
                elif d == config.DOWN and self.apple.y - y > 0 and not isNextMoveCollision(self, config.DOWN):
                    self.snake.moveDown()
                #if you can't move in the same direction, just pick one that brings you closer to the apple
                elif self.apple.x-x < 0 and d != config.RIGHT and not isNextMoveCollision(self, config.LEFT):  #Make sure snake isn't moving right
                    self.snake.moveLeft()
                elif self.apple.x-x > 0 and d != config.LEFT and not isNextMoveCollision(self, config.RIGHT): #Make sure snake isn't moving left
                    self.snake.moveRight()
                elif self.apple.y-y < 0 and d != config.DOWN and not isNextMoveCollision(self, config.UP): #Make sure snake isn't moving down
                    self.snake.moveUp()
                elif self.apple.y-y > 0 and d != config.UP and not isNextMoveCollision(self, config.DOWN): #Make sure snake isn't moving up
                    self.snake.moveDown()
                else:
                    if (d == config.LEFT or d == config.RIGHT) and not isNextMoveCollision(self, d + 2): #case when apple is directly behind snake
                        self.snake.direction = d + 2
                    elif (d == config.UP or d == config.DOWN) and not isNextMoveCollision(self, d - 2):
                        self.snake.direction = d - 2
                    else:
                        x = list(range(0,4))
                        random.shuffle(x)
                        for i in x + [4]: #iterate until you find a valid move
                            if i !=4 and not isNextMoveCollision(self,i):
                                self.snake.direction = i
                                break
                            if i == 4: #No move exists, move right
                                self.snake.direction = config.RIGHT

            else:
                # Interpret keystroke
                if keys[pygame.K_LEFT] and self.snake.last_moved != config.RIGHT:
                    self.snake.moveLeft()
                if keys[pygame.K_RIGHT] and self.snake.last_moved != config.LEFT:
                    self.snake.moveRight()
                if keys[pygame.K_DOWN] and self.snake.last_moved != config.UP:
                    self.snake.moveDown()
                if keys[pygame.K_UP] and self.snake.last_moved != config.DOWN:
                    self.snake.moveUp()

            if (keys[pygame.K_ESCAPE]):
                self._running = False

            self.on_loop()
            self.on_render()

            time.sleep((100.0 - config.SPEED) / 1000.0);

        self.on_cleanup()
        return self.snake.score()

if __name__ == "__main__" :
    theApp = App(len(sys.argv) > 1 and sys.argv[1] == 'ai')
    theApp.on_execute()
