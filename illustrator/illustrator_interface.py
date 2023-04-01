from abc import ABC, abstractmethod


class Illustrator(ABC):
    @abstractmethod
    def generate_image(self, prompt, customizer=None):
        pass