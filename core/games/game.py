import collections
from abc import abstractmethod, ABC

import numpy
from tqdm import tqdm

from core import SimpleAgent


class Game(ABC):
    """
    This is the abstract base class for games. The ``__init__`` method needs to be overwritten where a scores matrix
    and the environment size need to be created. These should then be passed into the ``__init__`` method of this ABC
    class. See the TrapGame for an example.

    Additionally, the get environment method need to be overwritten. See its documentation and TrapGame for an example.
    """

    @abstractmethod
    def __init__(self, scheduler, scores, environment_size, agents, *args, agent_shape=(5, 5), **kwargs):
        """
        :param scheduler: A Scheduler class.
        :param scores: The scores matrix. See TrapGame for an example.
        :param environment_size: The size of the environment, See TrapGame for  an example.
        :param agents: An integer representing the number of agents or a list of strings which are the agent names.
        :param agent_shape: The shape for building the SimpleAgents
        """
        # Put the reward grid in scores
        # opponent_action x my action
        self.scores = scores

        # todo add action mapping for string to int

        self.environment_size = environment_size  # This is the input size to the agent neural networks

        # everything from here on can be left as is

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

        # save the scheduler and load the agents
        self.scheduler = scheduler
        self.scheduler.load_agents(self.agents)

        # this maps agents to integers used for tracking performance
        self.agent_name_to_int = {agent.name: i for i, agent in enumerate(self.agents)}

        # This is where the accumulated rewards for each agent go. it is a matrix so you can track performance against
        # specific opponents
        self.rewards = numpy.zeros((len(self.agents), len(self.agents)))

        # this is where you can store the actions for each agent. just append to the inner lists
        self.actions = [[] for i in range(len(self.agents))]
        # This is where you can store the actions for each agent against a specific opponent again append to the inner
        # most list
        self.paired_actions = [[[] for j in range(len(self.agents))] for i in range(len(self.agents))]

    @abstractmethod
    def _get_environment(self, agent_1, agent_2):
        """
        This is where you compute the environment (the input) for each agent pair.

        :param agent_1: An agent
        :param agent_2: An agent
        :return: a pair of environment arrays
        """
        # build environment for each agent, the input to the network
        ...
        return

    def play(self, agent_1, agent_2):
        """
        This is the method for having two agents play each other where their aim is to maximise their own reward.

        :param agent_1: A SimpleAgent
        :param agent_2: A SimpleAgent
        """

        env1, env2 = self._get_environment(agent_1, agent_2)

        # get the predicted rewards for each action
        agent_1_predict = agent_1.predict(numpy.atleast_2d(env1))[0]
        agent_2_predict = agent_2.predict(numpy.atleast_2d(env2))[0]

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

    def run(self, rounds=10, **scheduler_kwargs):
        """
        This is the method for running a game.

        :param rounds: The number of rounds the game is to run for
        :param scheduler_kwargs: Any keyword arguments to be passed to the scheduler's get_round() method.
        """
        for _ in tqdm(range(rounds)):
            for agent1, agent2 in self.scheduler.get_round(**scheduler_kwargs):
                self.play(agent1, agent2)

    def get_score(self):
        return self.rewards.sum(1)

    def latest_actions(self, number_of_actions=20):
        return [collections.Counter(agent[-number_of_actions:]) for agent in self.actions]