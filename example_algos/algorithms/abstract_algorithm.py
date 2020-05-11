from abc import abstractmethod


class AbstractMOODAlgorimth:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def score_sample(self):
        pass

    @abstractmethod
    def score_pixels(self):
        pass
