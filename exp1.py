from matplotlib import pyplot as plt
from core import SafeGame

agents = ['Oliver', 'Harry', 'George', 'Noah', 'Jack', 'Ava', 'Emily', 'Isabella', 'Mia', 'Poppy',
          'Betsy', 'Charlie', 'Dave', 'Fiona', 'Kevin', 'Laura', 'Quentin', 'Rosy', 'Steve', 'Tracy']


game = SafeGame(agents=agents, agent_shape=(5, 5), random=True, know_opponent=False,
                money=1, trap_cost=0.1, hospital_cost=1.5)
game.run(4000)


game.plot_relative_rewards()
game.plot_all_agent_actions_through_time(80)
plt.show()

