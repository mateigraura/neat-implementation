from neat.reproduction import Reproduction
from neat.species import Species
from statistics import mean
from copy import deepcopy

fitness_funcs = {
    "max": max,
    "sum": sum,
    "mean": mean
}


class Population:
    def __init__(self, config):
        self.config = config

        self.reproduction = Reproduction(config)

        # might scale better for new funcs types, without adding cute ifs
        self.fitness_criterion = fitness_funcs[config.genome_params.fitness_criterion]

        self.population = self.reproduction.create()
        self.species = []  # list(Species)
        self.speciate()
        self.champion = None
        self.generation = 0

    def evaluate(self, evaluator, iterations=0):
        curr_iteration = 0
        while curr_iteration < iterations:
            curr_iteration += 1

            evaluator(self.population, self.config)

            best = None
            for genome in self.population:
                if best is None or genome.fitness > best.fitness:
                    best = genome

            if self.champion is None or best.fitness > self.champion.fitness:
                self.champion = best

            print("Best genome in generation {}, has fitness {}"
                  .format(self.generation, self.champion.fitness))

            if self.champion.fitness >= self.config.genome_params.fitness_threshold:
                termination_message(self.generation, self.champion.fitness)
                return

            try:
                self.population = self.reproduction.breed(self.species)
            except Exception:
                if self.config.genome_params.reset_extinct:
                    self.population = self.reproduction.create()
                else:
                    return

            self.speciate()
            self.generation += 1

    def speciate(self):
        compatibility_threshold = self.config.genome_params.compatibility_threshold
        species = []

        # assign members to each of the species
        population = deepcopy(self.population)
        for s in self.species:
            for idx, p in enumerate(population):
                distance = p.compatibility_distance(s.champion, p)
                if distance < compatibility_threshold:
                    population.pop(idx)
                    s.update(p)
            species.append(s)

        # divide newborn population or unspeciated individuals into species
        for p in population:
            distances = []
            for s in species:
                distance = p.compatibility_distance(s.champion, p)
                if distance < compatibility_threshold:
                    distances.append((distance, s))

            # check how py checks for empty list |if distances|,
            # might be slow if it computes len first
            if distances:
                _, s = min(distances, key=lambda d: d[0])
                s.update(p)
            else:
                s = Species()
                s.update(p)
                species.append(s)

        # this is obviously dumb here, will change
        for s in species:
            s.update_avg_fitness()
        self.species.clear()
        self.species = species


def termination_message(gen, fit):
    print("#######################################################################")
    print("Generation {} reached fitness termination threshold {}".format(gen, fit))
    print("#######################################################################")
