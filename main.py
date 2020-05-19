from Game.Snake import *
from Solvers.QLearning import *

snake = Snake()
# snake.run()
agent = QL(snake)
agent.solve()