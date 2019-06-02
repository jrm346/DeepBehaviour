import random

from .scheduler import Scheduler


class RandomScheduler(Scheduler):
    def __init__(self):
        self.agents = None
        super().__init__()

    def load_agents(self, agents: list):
        self.agents = agents[:]

    def get_round(self, **kwargs):
        random.shuffle(self.agents)
        return zip(self.agents[:int(len(self.agents) / 2)], self.agents[int(len(self.agents) / 2):])
