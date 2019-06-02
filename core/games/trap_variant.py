import numpy

from .game import Game


class TrapVariantGame(Game):
    """
    This is a variance on the trap game where your trap doesn't destroy your money. Additionally, stealing cause no
    depreciation.

        N       T                   S
    N   money   money-trap_cost     2*money
    T   money   money-trap_cost     money-hospital
    S   0       money-trap_cost     money
    """
    def __init__(self, scheduler, random=True, know_opponent=False, money=1, trap_cost=0.1, hospital_cost=1.5,
                 agents=10, **kwargs):

        # The first column are the rewards for doing nothing, second for trapping, third for stealing
        scores = numpy.array([[money,   money - trap_cost,  2 * money],
                              [money,   money - trap_cost,  money - hospital_cost],
                              [0,       money - trap_cost,  money]])

        # calculate the size of the environment
        number_of_agents = len(agents) if not isinstance(agents, int) else agents
        environment_size = 1 + number_of_agents if know_opponent else 1

        self.random = random
        self.know_opponents = know_opponent

        super().__init__(scheduler, scores=scores, environment_size=environment_size, agents=agents, **kwargs)

    def _get_environment(self, agent_1, agent_2):
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

        return env1, env2

