from Game.Snake import *
from Solvers.QLearning import *
from Solvers.DeepQLearning import *
snake = Snake()
# snake.run()
# agent = QL(snake)
agent = DQNAgent(snake)
agent.solve()