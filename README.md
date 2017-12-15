# CS 221 Project

To run, install `pygame` and then run: `python play_game.py`.
Control the snake using up/down/left/right, press `esc` to quit.
Every time you get an apple, your score increases by the amount printed to console.
When you die, your final score is printed.

The essential code for purposes of this project is as follows:

'learning.py' generally has all of our training code/tensorflow code, etc. It has all the code of processing replay memory, etc.

'play_game.py' is the main engine of the game. It's where the main method is and is the file that has the core functionality of the game. Anyone wanting to understand the code should look at the main method, as this is where a lot of the core functions are called.

'config.py' has most of our core parameters, such as number of neurons per layer, discount factor, speed of the game, etc. The only key parameters it doesn't have are some of the parameters which are set directly in the 'epsilon_greedy' function. For example, for a user to modify the decay schedule, they would need to go into and modify the 'epsilon_greedy' function directly. To slow down the speed of the game, simply go into the config file and adjust the SPEED constant to something between 0 and 100, with 100 being the fastest.

All other files are basic class file games that are necessary for the function of the game.

To run the game, simply type: 'python3 play_game.py -r 1 -a -n -d'. This will have the game run using the neurons. If you want to run the game 10 times, you would type 'python3 play_game.py -r 10 -a -n -d'. the '-d' flag causes the display to appear. Hence, when we were training, we would type something like 'python3 play_game.py -r 100000 -a -n -p'. If you're training from scratch, you want to make sure all the .pkl files are deleted, as well as the neural networks (my_dqn files). The '-p' flag causes the game to train. If you want to play the game by using arrow keys, type 'python3 play_game.py -r 1 -d'.
