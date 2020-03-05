from statistics import mean


class Species:
    def __init__(self):
        self.members = []
        self.avg_fitness = None
        self.champion = None
        self.staleness = 0

    def update(self, new_member):
        self.members.append(new_member)
        if self.champion is None or \
                new_member.fitness > self.champion.fitness:
            self.champion = new_member
        else:
            self.staleness += 1.5

    def update_avg_fitness(self):
        self.avg_fitness = mean([member.fitness for member in self.members])
