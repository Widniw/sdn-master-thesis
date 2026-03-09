from utils import json2networkx

class NetworkSimulator:
    def __init__(self, topology_path):
        self.G = json2networkx(topology_path)
        