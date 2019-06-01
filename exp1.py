from matplotlib import pyplot as plt
from core import SafeGame

agents = ['Oliver', 'Harry', 'George', 'Noah', 'Jack', 'Ava', 'Emily', 'Isabella', 'Mia', 'Poppy']


game = SafeGame(agents=agents, agent_shape=(5, 5), random=True, know_opponent=False,
                money=1, trap_cost=0.1, hospital_cost=1.5)
game.run(1000)


game.plot_latest_actions(100)
game.plot_total_reward()
game.plot_relative_rewards()
game.plot_all_agent_actions_through_time(20)
plt.show()

