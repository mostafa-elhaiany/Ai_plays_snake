# Ai plays Snake

Qlearning and Deep QLearning agents to solve a game of snake

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

The code runs on python3
you'll need the following libraries

```
pygame
```
which handles the game GUI

and 

```
numpy
```
which was used in Qlearning for building the qtable

```
keras, along with tensorflow backend
```
which was used in DQN for building and training the network


### Installing



make sure you have a python3 setup up and running

then to install the needed libraries

```
pip install pygame keras tensorflow numpy
```

to make sure everything is up and running

```
python main.py
```
this should start a game of snake where you can play it yourself 


### Break down into file system and Algorithms used

the code is divided into two parts, the game, and the solvers

```
GAME
```
for the game folder settings hold some of the constants and settings used for colours number of rows and columns etc,
Snake holds the code for building the game with pygame GUI, all the needed information are commented in the file

```
Solvers
```
for the solvers folder there is a different class for every solver,

1)  QLearning.py
        Qlearning is a smart algorithm that builds a table where given a certain observation you can get the best action you can do in that certain situation.

        Its basically a cheatsheet for your Ai. 

        So for snake this is a bit devious, you'd want to give the Ai ability to see everything on the board how ever that would require a huge Qtable since you need to store every possible combination of the board and you'll need the agent to play enough to build such a huge table with good values,
        so we need a simpler view of the world, the simpler world I've chosen was to tell the Agent whether or not there is an object in each of the possible 4 directions, along with whether the apple is infront of him or not and whether its in the upper direction or not,
        ofcourse this isn't enough for the snake to be able to play well enough however with the memory limitations its quite good.

        however, for the snake to be good it needs to see more of the world without having the memory limitations of having a huge cheatsheet, which brings us to --->


2)  DeepQLearning.py
        DQN Agents combine the cheatsheet of QL with the brains of ML and neural networks

        instead of having to limit the observation space to keep in a table the Neural network keeps up the information for us.
        And of course with the great ability to generalize and get better.
        Moreover, A neural network backpropagation handles the same learning rate and parts of the equations we use for the QL part.

        So, I started with the same thought of giving the network the entire view of the board so that it could see exactly what we see,
        so I used a convolutional neural network, which is a neural network that is applied on images, 
        the network takes the entire frame or the grid and calculates the qValues for the next action

        however since we train the model every step this gets really slow,
        which isn't so good since the Agent needs time to learn, so again I was forced to simplify the view of the world
        however this time to boost training speed,
        
        So, the first Idea was to give the snake ability to see the 20x20 image sorrounding his head, meaning his head was always in the middle of the screen,

        this boosted the speed by a bunch,
        however it was still kind of slow, and the network has to be a simple two layer network, which could have been better if it was a deeper/wider network

        So, I simplified the world a bit more going into only the sorrounding 9x9 pixels but also giving the snake information about the apple,

        it could see how far away and in which direction along the x-axis, and the same for the y-axis
        and then just for the sake of it I gave it the ecluidean distance.

        and finally it was playing, like you'd normaly play snake, he wasn't the best in the world but he was okay

    
    the policy and reward system was the same with both agents,
    a greedy epsilon policy was used, where we use an epsilon value to decide if we're using what we already know or trying something new, to battle the whole exploitation vs exploration part

    the reward system was simple,
    when playing snake, when the snake is small you wanna go to the apple as far as you can,
    then when the snake is bigger you want to be more accurate about it

    so for the first while while the score was still small, the closer it got to the apple he got a reward and the further he went he got a negative reward, up until he was long enough then the distance from the apple isn't a factor anymore

    the rest was simple, eat the apple take a positive reward, eat yourself or hit the wall you get a negative reward and you die

    which I see worked nicely.



### Running the Agents

in the main file,
comment the game.run() function and uncomment which agent you need to solve the game along with the agent.solve() function
and watch the snake go!





