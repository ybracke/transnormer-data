from abc import ABC, abstractmethod

class BaseMaker(ABC):

    @abstractmethod
    def make(self):
        """Abstract method for calling the maker and saving its results"""
        pass