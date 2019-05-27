import numpy
from core.simple_agent import SimpleAgent


class Game1:
    def __init__(self, random=True, agents=None, money=1, booby=0.1, hospital=-0.5):
        self.name = 'safes'
        # N = do nothing, B = boobytrap, S = steal
        # my_action x opponent_action
        # NN NB NS
        # BN BB BS
        # SN SB SS
        self.scores = numpy.array([[money,       money,          0],
                                   [money-booby, money-booby,    money-booby],
                                   [2*money,     hospital,       money]])
        self.scores.transpose()

        self.random = random
        self.agents = {agents[i].name: i for i in range(len(agents))} if agents else None

    def _get_environments(self, agent_1, agent_2):
        # build environment for each agent, the input to the network
        if self.random:
            env1 = numpy.random.rand(1)
            env2 = numpy.random.rand(1)
        elif self.agents:
            env1 = numpy.zeros(len(self.agents))
            env1[self.agents[agent_2.name]] = 1

            env2 = numpy.zeros(len(self.agents))
            env2[self.agents[agent_1.name]] = 1

            if self.random:
                env1 = numpy.concatenate([numpy.random.rand(1), env1])
                env2 = numpy.concatenate([numpy.random.rand(1), env2])
        else:
            env1 = numpy.zeros(1)
            env2 = numpy.zeros(1)

        return env1, env2

    def environment_size(self):
        if self.agents:
            return 1 + len(self.agents)
        else:
            return 1

    def rewards_size(self):
        return len(self.scores)

    def __call__(self, agent_1, agent_2):
        # type: (SimpleAgent, SimpleAgent) -> tuple

        env1, env2 = self._get_environments(agent_1, agent_2)

        # get the predicted rewards for each action
        agent_1_predict = agent_1.predict(env1)
        agent_2_predict = agent_2.predict(env2)

        # get the predicted best action
        agent_1_action = numpy.argmax(agent_1_predict)
        agent_2_action = numpy.argmax(agent_2_predict)

        # get the true rewards based on how the opponent acts
        agent_1_rewards = self.scores[agent_2_action]
        agent_2_rewards = self.scores[agent_1_action]

        # train on the true rewards
        agent_1.train(env1, agent_1_rewards)
        agent_2.train(env2, agent_2_rewards)

        # get and return the rewards for this game
        agent_1_reward = self.scores[agent_2_action][agent_1_action]
        agent_2_reward = self.scores[agent_1_action][agent_2_action]

        return agent_1_reward, agent_2_reward
