import collections

import math
import numpy
from matplotlib import pyplot as plt


def plot_latest_actions(game, number_of_actions=20):
    """
    Plots the latest n actions of all the agents. To get the plot to show


    :param game: A Game subclass
    :param number_of_actions: the number of actions desired for plotting
    :returns: a pyplot figure
    """
    fig = plt.figure()
    actions = game.latest_actions(number_of_actions)
    percentages = [[(agent[i] / sum(agent.values()) * 100.0) for i in range(3)] for agent in actions]
    trap = [agent[1] for agent in percentages]
    steal = [agent[2] for agent in percentages]
    nothing = [agent[0] for agent in percentages]
    agents = list(range(len(actions)))
    p1 = plt.bar(agents, steal)
    p2 = plt.bar(agents, trap, bottom=steal)
    p3 = plt.bar(agents, nothing, bottom=[sum(x) for x in zip(steal, trap)])

    plt.ylabel('Percentage of action')
    plt.title('Latest Actions')
    plt.xlabel('Agents')
    plt.xticks(list(range(len(game.agents))), [agent.name for agent in game.agents])
    plt.legend((p1[0], p2[0], p3[0]), ('Steal', 'Trap', 'Nothing'))

    return fig

def plot_relative_rewards(game):
    """
    Plots the difference in rewards between two agents. If the value is positive it means the row agent did better
    than the column agent

    :return:
    """
    plt.figure()
    relative_scores = numpy.tril(numpy.copy(game.rewards)) - numpy.fliplr(
        numpy.flipud(numpy.triu(numpy.copy(game.rewards))))
    plt.matshow(relative_scores)
    plt.xticks(list(range(len(relative_scores))), [agent.name for agent in game.agents])
    plt.yticks(list(range(len(relative_scores))), [agent.name for agent in game.agents])
    plt.colorbar()
    plt.title("Relative Rewards")


def plot_total_reward(game):
    """
    Plots the total reward of each agent

    :return:
    """
    plt.figure()
    plt.title("Total Reward")
    plt.bar(list(range(len(game.get_score()))), game.get_score())
    plt.xticks(list(range(len(game.agents))), [agent.name for agent in game.agents])


def plot_agent_actions_through_time(game, agent_name, splits):
    """
    Plot the actions of a single agent through time. splits time into n number of splits.

    :param agent_name: the agent to plot
    :param splits: the number of pieces to split time into
    :return:
    """
    plt.figure()
    actions = game.actions[game.agent_name_to_int[agent_name]]
    split_size = math.ceil(len(actions) / splits)
    split_counts = [collections.Counter(actions[i * split_size:(i + 1) * split_size]) for i in range(splits)]

    percentages = [[(split[i] / sum(split.values()) * 100.0) for i in range(3)] for split in split_counts]
    trap = [split[1] for split in percentages]
    steal = [split[2] for split in percentages]
    nothing = [split[0] for split in percentages]

    splits = list(range(splits))
    p1 = plt.bar(splits, steal)
    p2 = plt.bar(splits, trap, bottom=steal)
    p3 = plt.bar(splits, nothing, bottom=[sum(x) for x in zip(steal, trap)])

    xticks = [sum(split.values()) for split in split_counts]
    xticks = [f"{sum(xticks[:i])} to {sum(xticks[:i + 1])}" for i in range(len(xticks))]
    plt.ylabel('Percentage of action')
    plt.title(f'Agent {agent_name} actions through time')
    plt.xlabel('Actions in time')
    plt.xticks(splits, xticks)
    plt.legend((p1[0], p2[0], p3[0]), ('Steal', 'Trap', 'Nothing'))


def plot_all_agent_actions_through_time(game, splits):
    splits = list(range(splits))
    game.agents = sorted(game.agents, key=lambda x: x.name)
    fig = plt.figure()
    fig.suptitle("Percentage of acton taken by agents through time")
    trap_color = 'red'
    steal_color = 'blue'
    nothing_color = 'green'
    # plot each agent
    for i, agent in enumerate(game.agents):
        sub_plt = fig.add_subplot(len(game.agents) + 1, 1, i + 1)
        actions = game.actions[game.agent_name_to_int[agent.name]]
        split_size = math.ceil(len(actions) / len(splits))
        split_counts = [collections.Counter(actions[i * split_size:(i + 1) * split_size]) for i in range(len(splits))]

        percentages = [[(split[i] / sum(split.values()) * 100.0) for i in range(3)] for split in split_counts]
        trap = [split[1] for split in percentages]
        steal = [split[2] for split in percentages]
        nothing = [split[0] for split in percentages]

        p1 = sub_plt.bar(splits, steal, color=steal_color)
        p2 = sub_plt.bar(splits, trap, bottom=steal, color=trap_color)
        p3 = sub_plt.bar(splits, nothing, bottom=[sum(x) for x in zip(steal, trap)], color=nothing_color)

        sub_plt.set_ylabel(f'{agent.name}', rotation=0)
        sub_plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)

    # plot the population
    sub_plt = fig.add_subplot(len(game.agents) + 1, 1, len(game.agents) + 1)
    actions = []
    for round in numpy.transpose(numpy.array(game.actions)):
        actions.extend(round)

    split_size = math.ceil(len(actions) / len(splits))
    split_counts = [collections.Counter(actions[i * split_size:(i + 1) * split_size]) for i in range(len(splits))]

    percentages = [[(split[i] / sum(split.values()) * 100.0) for i in range(3)] for split in split_counts]
    trap = [split[1] for split in percentages]
    steal = [split[2] for split in percentages]
    nothing = [split[0] for split in percentages]

    p1 = sub_plt.bar(splits, steal, color=steal_color)
    p2 = sub_plt.bar(splits, trap, bottom=steal, color=trap_color)
    p3 = sub_plt.bar(splits, nothing, bottom=[sum(x) for x in zip(steal, trap)], color=nothing_color)

    sub_plt.set_ylabel("Population", rotation=0)
    sub_plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

    fig.legend((p1[0], p2[0], p3[0]), ('Steal', 'Trap', 'Nothing'))
