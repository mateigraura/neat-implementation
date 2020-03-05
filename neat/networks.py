from neat.node_gene import NodeTypes
from neat.activator import Activator
from statistics import mean
import random


# TODO : for mutate_add_node case, pin->out connection becomes disabled
# TODO : the weight of pin->hid and hid->out should be summed up (i think)
class FeedForwardNetwork:
    def __init__(self, config, genome):
        self.in_nodes = []
        self.out_nodes = []
        self.input_pairs = []
        self.connections = []
        self.activation_func = None
        self._map_genome(config, genome)

    def activate(self, inputs):
        for k, v in zip(self.in_nodes, inputs):
            self.input_pairs.append((k.key, v))

        results = []
        for in_node_key, weight in self.connections[:len(self.input_pairs)]:
            try:
                results.append(self.input_pairs[in_node_key - 1][1] * weight)
            except Exception:
                pass

        agg = sum(results)
        bias = random.random() * 1.0
        return self.activation_func(agg * bias)

    def _map_genome(self, config, genome):
        for node in genome.nodes:
            if node.node_type == NodeTypes.INPUT:
                self.in_nodes.append(node)
            elif node.node_type == NodeTypes.OUTPUT:
                self.out_nodes.append(node)

        self.connections = [(cg.in_node.key, cg.weight) for cg in genome.connection_genes
                            if cg.enabled]
        activator = Activator(config.genome_params.activation_func)
        self.activation_func = activator.get_func()
