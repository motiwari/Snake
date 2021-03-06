from random import randint
import pygame
import time
import apple
import snake
import state
import config
import gameengine
import sys
import random
import argparse
import pickle
import tensorflow as tf
import numpy as np
from collections import deque
import os
from learning import *

def isNextMoveCollision(pyg, direction):
    dummy_head = None
    x_change = {
        config.RIGHT: pyg.snake.step,
        config.LEFT: -pyg.snake.step,
        config.UP: 0,
        config.DOWN: 0,
    }
    y_change = {
        config.RIGHT: 0,
        config.LEFT: 0,
        config.UP: -pyg.snake.step,
        config.DOWN: pyg.snake.step,
    }
    dummy_head = snake.Head(pyg.snake.x[0] + x_change[direction], pyg.snake.y[0] + y_change[direction])

    # Check Board collision
    if dummy_head.x < 0 or dummy_head.x >= pyg.windowWidth or \
        dummy_head.y < 0 or dummy_head.y >= pyg.windowHeight:
        return True
    # Check Snake collision
    # Need to account for the fact that the snake will have moved by 1,
    # so we don't start on 3rd segment of snake
    for i in range(1, pyg.snake.length - 1):
        dummy_head2 = snake.Head(pyg.snake.x[i], pyg.snake.y[i])
        if pyg.gameEngine.isCollision(dummy_head, dummy_head2):
            return True
    return False

def baselineGreedy(pyg):

    x = pyg.snake.x[0]
    y = pyg.snake.y[0]
    d = pyg.snake.last_moved

    if pyg.apple.x - x < 0 and d != config.RIGHT and not isNextMoveCollision(pyg, config.LEFT):  # Make sure snake isn't moving right
        return config.LEFT
    elif pyg.apple.x - x > 0 and d != config.LEFT and not isNextMoveCollision(pyg, config.RIGHT): # Make sure snake isn't moving left
        return config.RIGHT
    elif pyg.apple.y - y < 0 and d != config.DOWN and not isNextMoveCollision(pyg, config.UP): # Make sure snake isn't moving down
        return config.UP
    elif pyg.apple.y - y > 0 and d != config.UP and not isNextMoveCollision(pyg, config.DOWN): # Make sure snake isn't moving up
        return config.DOWN
    else:
        # Case when apple is directly behind snake
        if (d == config.LEFT or d == config.RIGHT) and not isNextMoveCollision(pyg, d + 2):
            return d + 2
        elif (d == config.UP or d == config.DOWN) and not isNextMoveCollision(pyg, d - 2):
            return d - 2
        else:
            x = list(range(0,4))
            random.shuffle(x)
            for i in x: # Iterate until you find a valid move
                if not isNextMoveCollision(pyg, i):
                    return i

            return config.RIGHT
class App:
    windowWidth = config.WINDOW_WIDTH
    windowHeight = config.WINDOW_HEIGHT
    boardWidth = windowWidth/config.STEP_SIZE
    boardHeight = windowHeight/config.STEP_SIZE

    def __init__(self, args, sess):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.gameEngine = gameengine.GameEngine()
        self.snake = snake.Snake(3)
        self.apple = apple.Apple(5,5)
        self.apple.x, self.apple.y = random.choice(self.gameEngine.getBoardFreeSquares(self.snake))
        self.usingAI = args.ai
        self.usingNN = args.ain
        self.verbose = args.verbose
        self.display = args.display
        self.saveHistory = args.history
        self.history = []
        self.actionHistory = []
        self.sess = sess

    def on_init(self):
        pygame.init()
        if self.display == True:
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
        # Does snake collide with itself?

        for i in range(2, self.snake.length):
            # Fix this
            dummy_head = snake.Head(self.snake.x[i], self.snake.y[i])
            if self.gameEngine.isCollision(self.snake.head, dummy_head):
                self.snake.score -= 100
                self.snake.addActionAndReward(self.snake.direction, 0)
                if self.verbose:
                    print("You lose! Collision: ")
                    print("FINAL SCORE: ", self.snake.score)
                    print(self.snake.ars)
                self._running = False
                return
                #exit(0)

        if self.snake.head.x < 0 or self.snake.head.x >= self.windowWidth or \
            self.snake.head.y < 0 or self.snake.head.y >= self.windowHeight:
            #punish the player for running into itself
            self.snake.score -= 100
            self.snake.addActionAndReward(self.snake.direction, 0)
            if self.verbose:
                print("You lose! Off the board!")
                print("FINAL SCORE: ", self.snake.score)
                print(self.snake.ars)
            self._running = False
            return
            #exit(0)

        #reward the player for surviving longer
        # self.snake.score -= 1
        # Does snake eat apple?
        if self.gameEngine.isCollision(self.apple, self.snake.head):
            self.snake.length = self.snake.length + 1
            self.snake.score += self.apple.value
            if self.verbose:
                print("Ate apple with value ", self.apple.value)
            self.snake.addActionAndReward(self.snake.direction, self.apple.value)
            self.apple.value = 100
            freeSqs = self.gameEngine.getBoardFreeSquares(self.snake)
            if freeSqs == []:
                if self.verbose:
                    print("You WON Snake!!")
                    print("FINAL SCORE: ", self.snake.score)
                #exit(0)
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
                if(self.usingNN):
                    q_values = self.choose_ai_move(self.sess)
                else:
                    self.choose_ai_move()

            # Not using AI, Snake is player-controlled
            else:
                self.use_player_move(keys)

            if (keys[pygame.K_ESCAPE]):
                self._running = False

            #save game STATE

            if(self.saveHistory):
                self.history.append(state.State(self))
                self.actionHistory.append(self.snake.last_attempted_action)

            # Update the position and direction of snake
            # As well as value of apple
            self.on_loop()

            if args.display == True:
                self.on_render()

            time.sleep((100.0 - config.SPEED) / 1000.0);

        #pickle.dump(self.history,open('gamehistory.pkl','wb'))
        #save game STATE
        if(self.saveHistory):
            self.history.append(state.State(self))
            self.actionHistory.append(self.snake.last_attempted_action)

        self.on_cleanup()

        return self.history, self.actionHistory

    def use_player_move(self, keys):
        # Interpret keystroke
        if keys[pygame.K_LEFT]:
            self.snake.last_attempted_action = config.LEFT
            if self.snake.last_moved != config.RIGHT:
                self.snake.moveLeft()
        if keys[pygame.K_RIGHT]:
            self.snake.last_attempted_action = config.RIGHT
            if self.snake.last_moved != config.LEFT:
                self.snake.moveRight()
        if keys[pygame.K_DOWN]:
            self.snake.last_attempted_action = config.DOWN
            if self.snake.last_moved != config.UP:
                self.snake.moveDown()
        if keys[pygame.K_UP]:
            self.snake.last_attempted_action = config.UP
            if self.snake.last_moved != config.DOWN:
                self.snake.moveUp()

    def choose_ai_move(self,sess = None):
        x = self.snake.x[0]
        y = self.snake.y[0]
        d = self.snake.last_moved

        # Move closer to the apple
        if self.usingNN:
                # Online DQN evaluates what to do
                s = state.State(self)
                features = preprocess_observation(s)
                action = 0
                #with tf.Session() as sess:
                # step = global_step.eval()
                if self.verbose:
                    print(np.sum(features,axis=2))
                q_values = online_q_values.eval(feed_dict={X_state: [features]})
                #print(features)
                if self.verbose:
                    print(q_values)
                isOnEdge = s.isonedge()
                suggestedAction = baselineGreedy(self)
                action = epsilon_greedy(q_values, self.snake.length, isOnEdge, suggestedAction)
                    #CHECK TO MAKE SURE THAT CHOSEN DIRECTION IS VALID
                if self.verbose:
                    print(action)
                self.snake.last_attempted_action = action
                if d == config.RIGHT:
                    if action != config.LEFT:
                        self.snake.direction = action
                    else:
                        self.snake.direction = d
                elif d == config.LEFT:
                    if action != config.RIGHT:
                        self.snake.direction = action
                    else:
                        self.snake.direction = d
                elif d == config.UP:
                    if action != config.DOWN:
                        self.snake.direction = action
                    else:
                        self.snake.direction = d
                elif d == config.DOWN:
                    if action != config.UP:
                        self.snake.direction = action
                    else:
                        self.snake.direction = d

                return q_values
        else:
            if self.apple.x - x < 0 and d != config.RIGHT and not isNextMoveCollision(self, config.LEFT):  # Make sure snake isn't moving right
                self.snake.last_attempted_action = config.LEFT
                self.snake.moveLeft()
            elif self.apple.x - x > 0 and d != config.LEFT and not isNextMoveCollision(self, config.RIGHT): # Make sure snake isn't moving left
                self.snake.last_attempted_action = config.RIGHT
                self.snake.moveRight()
            elif self.apple.y - y < 0 and d != config.DOWN and not isNextMoveCollision(self, config.UP): # Make sure snake isn't moving down
                self.snake.last_attempted_action = config.UP
                self.snake.moveUp()
            elif self.apple.y - y > 0 and d != config.UP and not isNextMoveCollision(self, config.DOWN): # Make sure snake isn't moving up
                self.snake.last_attempted_action = config.DOWN
                self.snake.moveDown()
            else:
                # Case when apple is directly behind snake
                if (d == config.LEFT or d == config.RIGHT) and not isNextMoveCollision(self, d + 2):
                    self.snake.last_attempted_action = d + 2
                    self.snake.direction = d + 2
                elif (d == config.UP or d == config.DOWN) and not isNextMoveCollision(self, d - 2):
                    self.snake.last_attempted_action = d - 2
                    self.snake.direction = d - 2
                else:
                    x = list(range(0,4))
                    random.shuffle(x)
                    for i in x: # Iterate until you find a valid move
                        if not isNextMoveCollision(self, i):
                            self.snake.last_attempted_action = i
                            self.snake.direction = i
                            return

                    # No move exists, move right. THIS IS WHERE YOU DIE.
                    self.snake.last_attempted_action = config.RIGHT
                    self.snake.direction = config.RIGHT


def get_args(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    # For generating pairs of sites
    parser.add_argument('-a', '--ai', help='Use AI', action='store_true')
    parser.add_argument('-n', '--ain', help='Use AI NeuralNetwork', action='store_true') #if this flag is not set, it will default to baseline
    parser.add_argument('-d', '--display', help='Display the game graphically', action='store_true')
    parser.add_argument('-v', '--verbose', help='Verbose output', action='store_true')
    parser.add_argument('-p', '--history', help='Collect and Save State History', action='store_true')
    parser.add_argument('-r', '--runs', help='Number of runs for training of size batch size (see config)', type=int)
    args = parser.parse_args(arguments)
    return args

if __name__ == "__main__" :
    args = get_args(sys.argv[1:])

    with tf.Session() as sess:
        if os.path.isfile(cnfg.checkpoint_path + ".index"):
            saver.restore(sess, cnfg.checkpoint_path)
        else:
            init.run()
            copy_online_to_target.run()
        replay_memory_size = cnfg.replay_memory
        replay_memory = deque([], maxlen=replay_memory_size)
        final_scores = []
        if args.history:
            if os.path.isfile('./replay_memory.pkl'):
                replay_memory = pickle.load(open("replay_memory.pkl","rb"))
                print("loaded length")
                print(len(replay_memory))
            if os.path.isfile('./finalscores.pkl'):
                final_scores = pickle.load(open("finalscores.pkl","rb"))
        for i in range(args.runs):
            theApp = App(args, sess)
            stateHist, actionHist = theApp.on_execute()


            #this whole block of code takes care of loading game history and training the NN
            if args.history:
                final_scores.append(stateHist[-1].score)
                if args.verbose:
                    print("replay memory len")
                    #print(replay_memory)
                    print(len(replay_memory))

                gameHistory = pre_processHistory(stateHist, actionHist)
                for i, (a,b,c,d,e) in enumerate(gameHistory):
                #     print("state number: ", i + 1)
                #     size2 = cnfg.WIDTH_TILES * cnfg.HEIGHT_TILES
                #     a1 = 8 * a[:size2]
                #     a2 = 2 * a[size2:(2*size2)]
                #     a3 = a[(2*size2):(3*size2)]
                #     a4 = a[(3*size2):]
                #     # print(a1)
                #     # print(a2)
                #     # print(a1)
                #     # print(a2)
                #     aFinal = a1 + a2 + a3 + a4
                #     print("Old State:\n", aFinal.reshape(cnfg.WIDTH_TILES,cnfg.HEIGHT_TILES))
                    if args.verbose:
                        print("Direction", b)
                        print("Reward", c)
                #     d1 = 8 * d[:size2]
                #     d2 = 2* d[size2:(2*size2)]
                #     d3 = d[(2*size2):(3*size2)]
                #     d4 = d[(3*size2):]
                #     dFinal = d1 + d2 + d3 + d4
                #     print("New State:\n", dFinal.reshape(cnfg.WIDTH_TILES,cnfg.HEIGHT_TILES))
                #     print(e)
                #     print("\n")
                replay_memory.extend(gameHistory)
                if len(replay_memory) >= cnfg.training_start:
                    update(replay_memory, sess)
                if i % cnfg.save_steps == 0:

                    saver.save(sess, cnfg.checkpoint_path)
                    #give it time to save. i'm getting bugs
                    #time.sleep(50.0/1000.0)
                    pickle.dump(replay_memory,open('replay_memory.pkl','wb'))
                    pickle.dump(final_scores,open('finalscores.pkl','wb'))

                #this is a critical piece of code! do not delete unless you know what you're doing
                if i % cnfg.copy_steps == 0:
                    copy_online_to_target.run()

        if args.history:
            pickle.dump(replay_memory,open('replay_memory.pkl','wb'))
            pickle.dump(final_scores,open('finalscores.pkl','wb'))
