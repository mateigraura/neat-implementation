from neat.config import Config
from neat.population import Population
from neat.networks import FeedForwardNetwork
import os

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
outputs = [(0,), (1,), (1, 0), (0,)]


def get_config():
    curr_dir = os.path.dirname(__file__)
    file_path = os.path.join(curr_dir, "xor_config.json")
    return Config(file_path)


def eval_xor(genomes, config):
    for genome in genomes:
        genome.fitness = 4.0
        network = FeedForwardNetwork(config, genome)
        for _in, _out in zip(inputs, outputs):
            res = network.activate(_in)
            genome.fitness -= (res - _out[0]) ** 2


def execute(config):
    population = Population(config)
    population.evaluate(eval_xor, 600)


if __name__ == "__main__":
    execute(get_config())
