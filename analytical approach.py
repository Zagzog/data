# -*- coding:utf-8 -*-

import networkx as nx
import random
from main_program_support import opinion_start3 as os3
from main_program_support import degree_node
import math
import os
import copy

# define nodes state types
states = ['A', 'U']


class IoDynamics_modify(object):

    def __init__(self, information_layer, opinion_layer, source_node, alpha_a, frontier_extreme_number,
                 central_extreme_number, omega_w, central_opponents_number, central_supporters_number,
                 frontier_supporters_number):
        self.layer1 = information_layer
        self.layer2 = opinion_layer
        self.alpha = alpha_a
        self.fen = frontier_extreme_number
        self.cen = central_extreme_number
        self.ow = omega_w
        self.U = []
        self.A = []
        self.can_spread_A = []
        self.nodes_U = []
        self.nodes_A = []
        self.node_states = {}
        self.opinion = {}
        self.layer1_supporters_neighbors = {}
        self.layer2_supporters_neighbors = {}
        self.layer1_opponents_neighbors = {}
        self.layer2_opponents_neighbors = {}
        self.neighbor_of_source = []
        self.ow_nodes = {}
        self.source_nodes = source_node
        self.m = 0
        self.all_opinion = {}

        self.stubborn_supporters = []
        self.stubborn_opponents = []
        self.first_ten_opinion = {}
        self.con = int(central_opponents_number)
        self.csn = int(central_supporters_number)
        self.fsn = int(frontier_supporters_number)
        for nodes in self.layer1.nodes():
            self.first_ten_opinion[nodes] = []

        # set extreme nodes
        self.min_degree, self.max_degree = degree_node(self.layer2, self.fen, self.cen)
        self.remain_max_degree = []
        for nd1 in list(random.sample(self.max_degree, self.con)):
            self.stubborn_opponents.append(nd1)
        for md1 in self.max_degree:
            if md1 not in self.stubborn_opponents:
                self.remain_max_degree.append(md1)
        for nd_1 in list(random.sample(self.remain_max_degree, self.csn)):
            self.stubborn_supporters.append(nd_1)

        for nd2 in list(random.sample(self.min_degree, self.fsn)):
            self.stubborn_supporters.append(nd2)

        # initial opinion
        temporary_node = []
        for node2 in self.layer2.nodes():
            if node2 not in self.stubborn_supporters and node2 not in self.stubborn_opponents:
                temporary_node.append(node2)
        self.initiation_opinion = os3(temporary_node)

    def start1(self):
        for node2 in self.layer2.nodes():
            self.ow_nodes[node2] = self.ow
            self.opinion[node2] = 'Null'
            self.node_states[node2] = states[1]
            self.nodes_U.append(node2)
        for nodes1 in self.stubborn_supporters:
            self.initiation_opinion[nodes1] = 1
        for nodes2 in self.stubborn_opponents:
            self.initiation_opinion[nodes2] = -1
        self.U.append(len(self.nodes_U))
        self.A.append(len(self.nodes_A))
        self.node_states[self.source_nodes] = states[0]
        self.opinion[self.source_nodes] = self.initiation_opinion[self.source_nodes]
        self.nodes_U.remove(self.source_nodes)
        self.nodes_A.append(self.source_nodes)
        for supporter_node in self.stubborn_supporters:
            self.layer1_supporters_neighbors[supporter_node] = list(self.layer1.neighbors(supporter_node))
            self.layer2_supporters_neighbors[supporter_node] = list(self.layer2.neighbors(supporter_node))
        for opponent_node in self.stubborn_opponents:
            self.layer1_opponents_neighbors[opponent_node] = list(self.layer1.neighbors(opponent_node))
            self.layer2_opponents_neighbors[opponent_node] = list(self.layer2.neighbors(opponent_node))
        self.spread()
        return self.A, self.U, self.opinion, self.layer1_supporters_neighbors, self.layer2_supporters_neighbors, \
               self.layer1_opponents_neighbors, self.layer2_opponents_neighbors, self.source_nodes, self.nodes_A, \
               self.first_ten_opinion

    def spread(self):
        self.m += 1
        # initial information spread
        store_opinion1 = {}
        store_ow_number = {}
        store_nodes_U = []
        store_nodes_state = {}
        for node1 in self.layer1.nodes():
            store_ow_number[node1] = self.ow_nodes[node1]
            store_opinion1[node1] = self.opinion[node1]
            store_nodes_state[node1] = self.node_states[node1]
        for nodes2 in self.nodes_U:
            account1 = 0
            for neighbor1 in list(self.layer1.neighbors(nodes2)):
                if self.node_states[neighbor1] == states[0]:
                    if self.ow_nodes[neighbor1] != 0:
                        r1 = random.random()
                        spread_probability = (2*self.alpha / (1 + math.exp(-abs(self.opinion[neighbor1])))) * 1.0
                        if r1 <= spread_probability:
                            account1 += 1
            if account1 != 0:
                store_opinion1[nodes2] = self.initiation_opinion[nodes2]
                store_nodes_state[nodes2] = states[0]
                store_nodes_U.append(nodes2)
        for nodes3 in self.nodes_A:
            if self.ow_nodes[nodes3] != 0:
                store_ow_number[nodes3] -= 1
        for nodes4 in store_nodes_U:
            self.nodes_A.append(nodes4)
            self.nodes_U.remove(nodes4)
        for nodes5, value5 in store_ow_number.items():
            self.ow_nodes[nodes5] = value5
        for nodes6, value6 in store_nodes_state.items():
            self.node_states[nodes6] = value6
        for nodes7, value7 in store_opinion1.items():
            self.opinion[nodes7] = value7
            if self.ow_nodes[nodes7] != 0 and self.node_states[nodes7] == states[0]:
                self.first_ten_opinion[nodes7].append(value7)
        self.A.append(len(self.nodes_A))
        self.U.append(len(self.nodes_U))

        # opinion interaction
        store_opinion2 = {}
        for nodes3 in self.nodes_A:
            if nodes3 not in self.stubborn_supporters and nodes3 not in self.stubborn_opponents:
                neighbor_of_A = []
                for neighbor2 in self.layer2.neighbors(nodes3):
                    if self.opinion[neighbor2] != 'Null':
                        neighbor_of_A.append(neighbor2)
                if len(neighbor_of_A) != 0:
                    neighbor_influence = [self.opinion[neighbor3] - self.opinion[nodes3] for neighbor3 in neighbor_of_A]
                    opinion_new = (sum(neighbor_influence) * 1.0) / (len(neighbor_of_A) * 1.0) + self.opinion[nodes3]
                    store_opinion2[nodes3] = opinion_new
                else:
                    store_opinion2[nodes3] = self.opinion[nodes3]
            else:
                store_opinion2[nodes3] = self.opinion[nodes3]
        for nodes2 in self.nodes_A:
            self.opinion[nodes2] = store_opinion2[nodes2]

        # whether the information dissemination stops
        n_1 = 0
        for nodes3 in self.nodes_A:
            if nodes3 not in self.can_spread_A:
                n_2 = 0
                for node4 in self.layer1.neighbors(nodes3):
                    if self.node_states[node4] == states[1]:
                        n_2 += 1
                if n_2 == 0:
                    self.can_spread_A.append(nodes3)
        for nodes3 in self.nodes_A:
            if nodes3 not in self.can_spread_A:
                if self.ow_nodes[nodes3] != 0:
                    n_1 += 1
        if n_1 == 0:
            self.opinion_process()
        else:
            self.spread()

    def opinion_process(self):
        while self.m <= 2000:
            self.m += 1
            store_opinion3 = {}
            for nodes3 in self.nodes_A:
                if nodes3 not in self.stubborn_supporters and nodes3 not in self.stubborn_opponents:
                    neighbor_of_A = []
                    for nodes in self.layer2.neighbors(nodes3):
                        if self.opinion[nodes] != 'Null':
                            neighbor_of_A.append(nodes)
                    if len(neighbor_of_A) != 0:
                        neighbor_influence = [(self.opinion[nodes4] - self.opinion[nodes3]) * 1.0 for nodes4 in
                                              neighbor_of_A]
                        opinion_new = (sum(neighbor_influence) * 1.0) / (len(neighbor_of_A) * 1.0) + self.opinion[
                            nodes3] * 1.0
                        store_opinion3[nodes3] = opinion_new
                    else:
                        store_opinion3[nodes3] = self.opinion[nodes3]
                else:
                    store_opinion3[nodes3] = self.opinion[nodes3]
            for nodes3 in self.nodes_A:
                self.opinion[nodes3] = store_opinion3[nodes3]


def run_dynamics_modify(information_layer, opinion_layer, alpha_a, frontier_extreme_number, central_extreme_number,
                        omega_w, iteration_time, name, central_opponents_number, central_supporters_number,
                        frontier_supporters_number):
    G1 = information_layer
    G2 = opinion_layer
    alpha = alpha_a
    fen = frontier_extreme_number
    cen = central_extreme_number
    ow = omega_w
    mm = iteration_time
    con = central_opponents_number
    csn = central_supporters_number
    fsn = frontier_supporters_number
    source_list = list(nx.extrema_bounding(G1, compute='center'))

    path1 = 'E:/paper-data/analytical approach/' + name
    for i in range(mm):
        folder = os.path.exists(path1)
        if not folder:
            os.makedirs(path1)
    source_node1 = source_list[0]
    ide = IoDynamics_modify(G1, G2, source_node1, alpha, fen, cen, ow, con, csn, fsn)
    number_of_A = {}
    steady_opinion = {}
    for nodes in G2.nodes():
        number_of_A[nodes] = 0
        steady_opinion[nodes] = []

    for s1 in range(mm):
        ide2 = copy.deepcopy(ide)
        A_population, U_population, final_opinion, layer1_supporters_neighbors, layer2_supporters_neighbors, \
        layer1_opponents_neighbors, layer2_opponents_neighbors, source, nodes_A, first_ten_opinion = ide2.start1()

        for nodes in G2.nodes():
            if nodes in nodes_A:
                number_of_A[nodes] += 1
            if final_opinion[nodes] != 'Null':
                steady_opinion[nodes].append(final_opinion[nodes])

    g5 = open('E:/paper-data/analytical approach/' + name + '/average_opinion.txt', 'w')
    for keys2 in steady_opinion:
        if len(steady_opinion[keys2]) != 0:
            average_opinion = (sum(steady_opinion[keys2]) * 1.0) / (len(steady_opinion[keys2]) * 1.0)
            g5.write(str(keys2) + ':' + str(average_opinion) + '\n')
    g5.close()

    nn = 0
    for keys1 in number_of_A:
        if number_of_A[keys1] == 0:
            nn += 1
    if nn != 0:
        print("Not all nodes have 'A' state at least once")
    else:
        print("All nodes have 'A' state at least once")


def control_scheme(layer1, layer2, name, ii):
    print(ii)
    G1 = layer1
    G2 = layer2
    alpha1 = 0.05
    central_number = 10
    frontier_number = 100
    ow = 10
    m = 100
    con1 = 10
    csn1 = 0
    fsn1 = ii * 10
    nnnn = '/control-scheme/' + name + str(fsn1) + '-frontier extreme supporter'
    run_dynamics_modify(G1, G2, alpha1, frontier_number, central_number, ow, m, nnnn, con1, csn1, fsn1)


def experimental_scheme(layer1, layer2, name, ii):
    print(ii)
    G1 = layer1
    G2 = layer2
    alpha1 = 0.05
    central_number = 60
    frontier_number = 60
    ow = 10
    m = 100
    con1 = 10
    csn1 = ii
    fsn1 = ii
    nnnn = '/experimental-scheme/' + name + str(fsn1) + '-frontier and central extreme supporter'
    run_dynamics_modify(G1, G2, alpha1, frontier_number, central_number, ow, m, nnnn, con1, csn1, fsn1)


if __name__ == '__main__':
    G1 = nx.read_edgelist('E:/paper-data/network data/BA1_10000')
    G2 = nx.read_edgelist('E:/paper-data/network data/ER1_10000')
    for i in range(11):
        control_scheme(G1, G2, 'BA-ER', i*10)
    for i in range(51):
        experimental_scheme(G1, G2, 'BA-ER', i)