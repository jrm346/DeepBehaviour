from matplotlib import pyplot as plt
from core.games import TrapGame
from core.schedulers import RandomScheduler
from core.plotting import plot_all_agent_actions_through_time

agents = ['Oliver', 'Harry', 'George', 'Noah', 'Jack', 'Ava', 'Emily', 'Isabella', 'Mia', 'Poppy',
          'Betsy', 'Charlie', 'Dave', 'Fiona', 'Kevin', 'Laura', 'Quentin', 'Rosy', 'Steve', 'Tracy']

schedule = RandomScheduler()
game = TrapGame(scheduler=schedule, agents=agents, random=True, know_opponent=False,)
game.run(4000)


plot_all_agent_actions_through_time(game, 80)
plt.show()

