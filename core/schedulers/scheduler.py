from abc import ABC, abstractmethod


class Scheduler(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abstractmethod
    def load_agents(self, agents):
        """
        This is where you load the agents in to what ever structure needed for scheduling

        :param agents: A list of Agent objects
        """
        ...

    @abstractmethod
    def get_round(self, **kwargs):
        """
        This is the function used to provide the pairs of agents that will play the game for that round.

        :param kwargs: Any additional keyword arguments required
        :return: an iterable of agent pairs
        """
        ...