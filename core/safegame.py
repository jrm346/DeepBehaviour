import collections
import random
from tqdm import tqdm

import math
from matplotlib import pyplot as plt
import numpy
from core.simple_agent import SimpleAgent


class SafeGame:
    def __init__(self, random=True, agents=10, know_opponent=False, money=1, trap_cost=0.1, hospital_cost=0.5, agent_shape=(5, 5)):


        plt.style.use('seaborn')

        self.name = 'safes'
        # N = do nothing, B = boobytrap, S = steal
        # my_action x opponent_action
        # NN NB NS
        # BN BB BS
        # SN SB SS
        self.scores = numpy.array([[money,       money,          0],
                                   [money - trap_cost, money - trap_cost, money - trap_cost],
                                   [2 * money, money - hospital_cost, money]])
        self.scores = self.scores.transpose()

        self.random = random
        self.know_opponents = know_opponent

        number_of_agents = len(agents) if not isinstance(agents, int) else agents
        self.environment_size = 1 + number_of_agents if know_opponent else 1

        if isinstance(agents, int):
            if agents % 2 != 0:
                raise Exception("needs an even amount of agents")
            else:
                self.agents = [
                    SimpleAgent(str(i), self.environment_size, len(self.scores), agent_shape[0], agent_shape[1]) for i
                    in range(agents)]
        else:
            if len(agents) % 2 != 0:
                raise Exception("needs an even amount of agents")
            else:
                self.agents = [
                    SimpleAgent(str(name), self.environment_size, len(self.scores), agent_shape[0], agent_shape[1])
                    for name in agents]

        self.agent_name_to_int = {agent.name: i for i, agent in enumerate(self.agents)}
        self.rewards = numpy.zeros((len(self.agents), len(self.agents)))
        self.paired_actions = [[[] for j in range(len(self.agents))] for i in range(len(self.agents))]
        self.actions = [[]for i in range(len(self.agents))]

    def _get_environments(self, agent_1, agent_2):
        # build environment for each agent, the input to the network
        if self.random:
            env1 = numpy.random.rand(1)
            env2 = numpy.random.rand(1)
        else:
            env1 = numpy.zeros(1)
            env2 = numpy.zeros(1)
        if self.know_opponents:
            extra1 = numpy.zeros(len(self.agents))
            extra1[self.agent_name_to_int[agent_2.name]] += 1

            extra2 = numpy.zeros(len(self.agents))
            extra2[self.agent_name_to_int[agent_1.name]] += 1

            env1 = numpy.concatenate([env1, extra1])
            env2 = numpy.concatenate([env2, extra2])

        return numpy.atleast_2d(env1), numpy.atleast_2d(env2)

    def rewards_size(self):
        return len(self.scores)

    def play(self, agent_1, agent_2):
        # type: (SimpleAgent, SimpleAgent) -> tuple

        env1, env2 = self._get_environments(agent_1, agent_2)

        # get the predicted rewards for each action
        agent_1_predict = agent_1.predict(env1)[0]
        agent_2_predict = agent_2.predict(env2)[0]

        # get the predicted best action
        agent_1_action = numpy.argmax(agent_1_predict)
        agent_2_action = numpy.argmax(agent_2_predict)

        # get the true rewards based on how the opponent acts
        agent_1_rewards = numpy.atleast_2d(self.scores[agent_2_action])
        agent_2_rewards = numpy.atleast_2d(self.scores[agent_1_action])

        # train on the true rewards
        agent_1.train(env1, agent_1_rewards)
        agent_2.train(env2, agent_2_rewards)

        # get and return the rewards for this game
        agent_1_reward = self.scores[agent_2_action][agent_1_action]
        agent_2_reward = self.scores[agent_1_action][agent_2_action]

        # save results
        agent_1_index = self.agent_name_to_int[agent_1.name]
        agent_2_index = self.agent_name_to_int[agent_2.name]
        self.rewards[agent_1_index][agent_2_index] += agent_1_reward
        self.rewards[agent_2_index][agent_1_index] += agent_2_reward
        self.paired_actions[agent_1_index][agent_2_index].append(int(agent_1_action))
        self.paired_actions[agent_2_index][agent_1_index].append(int(agent_2_action))
        self.actions[agent_1_index].append(int(agent_1_action))
        self.actions[agent_2_index].append(int(agent_2_action))

    def make_round(self):
        random.shuffle(self.agents)
        return zip(self.agents[:int(len(self.agents)/2)], self.agents[int(len(self.agents)/2):])

    def run(self, rounds=10):
        for _ in tqdm(range(rounds)):
            round = list(self.make_round())
            for agent1, agent2 in round:
                self.play(agent1, agent2)

    def get_score(self):
        return self.rewards.sum(1)

    def latest_actions(self, number_of_actions=20):
        return [collections.Counter(agent[-number_of_actions:]) for agent in self.actions]

    def plot_latest_actions(self, number_of_actions=20):
        """
        Plots the latest n actions of all the agents.

        :param number_of_actions: the number of actions desired for plotting
        """
        plt.figure()
        actions = self.latest_actions(number_of_actions)
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
        plt.xticks(list(range(len(self.agents))), [agent.name for agent in self.agents])
        plt.legend((p1[0], p2[0], p3[0]), ('Steal', 'Trap', 'Nothing'))

    def plot_relative_rewards(self):
        """
        Plots the difference in rewards between two agents. If the value is positive it means the row agent did better
        than the column agent

        :return:
        """
        plt.figure()
        relative_scores = numpy.tril(numpy.copy(self.rewards)) - numpy.fliplr(
            numpy.flipud(numpy.triu(numpy.copy(self.rewards))))
        plt.matshow(relative_scores)
        plt.xticks(list(range(len(relative_scores))), [agent.name for agent in self.agents])
        plt.yticks(list(range(len(relative_scores))), [agent.name for agent in self.agents])
        plt.colorbar()
        plt.title("Relative Rewards")

    def plot_total_reward(self):
        """
        Plots the total reward of each agent

        :return:
        """
        plt.figure()
        plt.title("Total Reward")
        plt.bar(list(range(len(self.get_score()))), self.get_score())
        plt.xticks(list(range(len(self.agents))), [agent.name for agent in self.agents])

    def plot_agent_actions_through_time(self, agent_name, splits):
        """
        Plot the actions of a single agent through time. splits time into n number of splits.

        :param agent_name: the agent to plot
        :param splits: the number of pieces to split time into
        :return:
        """
        plt.figure()
        actions = self.actions[self.agent_name_to_int[agent_name]]
        split_size = math.ceil(len(actions)/splits)
        split_counts = [collections.Counter(actions[i*split_size:(i+1)*split_size]) for i in range(splits)]


        percentages = [[(split[i] / sum(split.values()) * 100.0) for i in range(3)] for split in split_counts]
        trap = [split[1] for split in percentages]
        steal = [split[2] for split in percentages]
        nothing = [split[0] for split in percentages]

        splits = list(range(splits))
        p1 = plt.bar(splits, steal)
        p2 = plt.bar(splits, trap, bottom=steal)
        p3 = plt.bar(splits, nothing, bottom=[sum(x) for x in zip(steal, trap)])

        xticks = [sum(split.values()) for split in split_counts]
        xticks = [f"{sum(xticks[:i])} to {sum(xticks[:i+1])}" for i in range(len(xticks))]
        plt.ylabel('Percentage of action')
        plt.title(f'Agent {agent_name} actions through time')
        plt.xlabel('Actions in time')
        plt.xticks(splits, xticks)
        plt.legend((p1[0], p2[0], p3[0]), ('Steal', 'Trap', 'Nothing'))

    def plot_all_agent_actions_through_time(self, splits):
        splits = list(range(splits))
        self.agents = sorted(self.agents, key=lambda x: x.name)
        fig = plt.figure()
        fig.suptitle("Percentage of acton taken by agents through time")
        trap_color = 'red'
        steal_color = 'blue'
        nothing_color = 'green'
        # plot each agent
        for i, agent in enumerate(self.agents):

            sub_plt = fig.add_subplot(len(self.agents)+1, 1, i+1)
            actions = self.actions[self.agent_name_to_int[agent.name]]
            split_size = math.ceil(len(actions)/len(splits))
            split_counts = [collections.Counter(actions[i*split_size:(i+1)*split_size]) for i in range(len(splits))]

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

        #plot the population
        sub_plt = fig.add_subplot(len(self.agents)+1, 1, len(self.agents)+1)
        actions = []
        for round in numpy.transpose(numpy.array(self.actions)):
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
