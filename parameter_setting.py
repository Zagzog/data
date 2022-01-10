# -*- coding:utf-8 -*-
"""
created by zazo
"""

import networkx as nx
import random
from main_program_support import opinion_start2 as os2
from main_program_support import opinion_start3 as os3
from main_program_support import opinion_start4 as os4
from main_program_support import degree_node
import math
import os
import copy

# define nodes state types
states = ['A', 'U']


# information-opinion dynamics
class IoDynamics_opinion_distribution(object):

    def __init__(self, information_layer, opinion_layer, source_node, alpha_a, frontier_extreme_number,
                 central_extreme_number, omega_w, central_supporters_number, distribution_type):
        self.layer1 = information_layer
        self.layer2 = opinion_layer
        self.alpha = alpha_a
        self.cen = central_extreme_number
        self.fen = frontier_extreme_number
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
        self.csn = int(central_supporters_number)
        for nodes in self.layer1.nodes():
            self.first_ten_opinion[nodes] = []

        # set extreme nodes
        self.min_degree, self.max_degree = degree_node(self.layer2, self.fen, self.cen)

        for nd1 in list(random.sample(self.min_degree, self.csn)):
            self.stubborn_opponents.append(nd1)
        for md1 in self.min_degree:
            if md1 not in self.stubborn_opponents:
                self.stubborn_supporters.append(md1)
        for nd2 in list(random.sample(self.max_degree, self.csn)):
            self.stubborn_supporters.append(nd2)
        for md2 in self.max_degree:
            if md2 not in self.stubborn_supporters:
                self.stubborn_opponents.append(md2)

        # initial opinion
        temporary_node = []
        for node2 in self.layer2.nodes():
            if node2 not in self.stubborn_supporters and node2 not in self.stubborn_opponents:
                temporary_node.append(node2)
        if distribution_type == 'normal':
            self.initiation_opinion = os4(temporary_node)
        elif distribution_type == 'power-law':
            self.initiation_opinion = os2(temporary_node)
        else:
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


def run_dynamics_opinion_distribution(information_layer, opinion_layer, alpha_a, frontier_extreme_number,
                                      central_extreme_number, omega_w, iteration_time, name, central_supporters_number,
                                      distribution_type):
    G1 = information_layer
    G2 = opinion_layer
    alpha = alpha_a
    cen = central_extreme_number
    fen = frontier_extreme_number
    ow = omega_w
    mm = iteration_time
    csn = central_supporters_number
    dt = distribution_type
    source_list = list(nx.extrema_bounding(G1, compute='center'))

    path1 = 'E:/paper-data/parameter_setting/initial opinion distribution/' + str(dt) + '/' + name
    for i in range(mm):
        folder = os.path.exists(path1 + '/number' + str(i))
        if not folder:
            os.makedirs(path1 + '/number' + str(i))
    source_node1 = source_list[0]
    ide = IoDynamics_opinion_distribution(G1, G2, source_node1, alpha, fen, cen, ow, csn, dt)
    number_of_A = {}
    steady_opinion = {}
    for nodes in G2.nodes():
        number_of_A[nodes] = 0
        steady_opinion[nodes] = []

    for s1 in range(mm):
        ide2 = copy.deepcopy(ide)
        A_population, U_population, final_opinion, layer1_supporters_neighbors, layer2_supporters_neighbors, \
        layer1_opponents_neighbors, layer2_opponents_neighbors, source, nodes_A, first_ten_opinion = ide2.start1()
        b = 'E:/paper-data/parameter_setting/initial opinion distribution/' + str(dt) + '/' + name + \
            '/number' + str(s1) + '/'
        g1 = open(b + 'population of knowing news.txt', 'w')
        for i_1 in A_population:
            g1.write(str(i_1) + '\n')
        g1.close()
        for nodes in G2.nodes():
            if nodes in nodes_A:
                number_of_A[nodes] += 1
            if final_opinion[nodes] != 'Null':
                steady_opinion[nodes].append(final_opinion[nodes])

        f1 = open(b + 'source_node.txt', 'w')
        f1.write(str(source))
        f1.close()

        f = open(b + 'opinion before silent.txt', 'w')
        for x1 in first_ten_opinion.keys():
            f.write(str(x1) + ':' + str(first_ten_opinion[x1]) + '\n')
        f.close()

        f2 = open(b + 'information layer-extreme supporters neighbors.txt', 'w')
        for m1 in layer1_supporters_neighbors.keys():
            f2.write(str(m1) + ':' + str(layer1_supporters_neighbors[m1]) + '\n')
        f2.close()
        f3 = open(b + 'opinion layer-extreme supporters neighbors.txt', 'w')
        for m2 in layer2_supporters_neighbors.keys():
            f3.write(str(m2) + ':' + str(layer2_supporters_neighbors[m2]) + '\n')
        f3.close()
        f4 = open(b + 'information layer-extreme opponents neighbors.txt', 'w')
        for m3 in layer1_opponents_neighbors.keys():
            f4.write(str(m3) + ':' + str(layer1_opponents_neighbors[m3]) + '\n')
        f4.close()
        f5 = open(b + 'opinion layer-extreme opponents neighbors.txt', 'w')
        for m4 in layer2_opponents_neighbors.keys():
            f5.write(str(m4) + ':' + str(layer2_opponents_neighbors[m4]) + '\n')
        f5.close()

        f6 = open(b + 'information layer-source neighbors.txt', 'w')
        for m5 in G1.neighbors(source_node1):
            f6.write(str(m5) + '\n')
        f6.close()
        f7 = open(b + 'opinion layer-source neighbors.txt', 'w')
        for m6 in G2.neighbors(source_node1):
            f7.write(str(m6) + '\n')
        f7.close()

    f = open('E:/paper-data/parameter_setting/initial opinion distribution/' + str(dt) + '/' + name
             + '/probability of knowing news.txt', 'w')
    for keys1 in number_of_A:
        f.write(str(keys1) + ':' + str(number_of_A[keys1]) + '\n')
    f.close()
    g5 = open('E:/paper-data/parameter_setting/initial opinion distribution/' + str(dt) + '/' + name
              + '/average_opinion.txt', 'w')
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


class IoDynamics_limit_dissemination(object):

    def __init__(self, information_layer, opinion_layer, source_node, alpha_a, frontier_extreme_number,
                 central_extreme_number, omega_w, central_supporters_number):
        self.layer1 = information_layer
        self.layer2 = opinion_layer
        self.alpha = alpha_a
        self.cen = central_extreme_number
        self.fen = frontier_extreme_number
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
        self.csn = int(central_supporters_number)
        for nodes in self.layer1.nodes():
            self.first_ten_opinion[nodes] = []

        # set extreme nodes
        self.min_degree, self.max_degree = degree_node(self.layer2, self.fen, self.cen)

        for nd1 in list(random.sample(self.min_degree, self.csn)):
            self.stubborn_opponents.append(nd1)
        for md1 in self.min_degree:
            if md1 not in self.stubborn_opponents:
                self.stubborn_supporters.append(md1)
        for nd2 in list(random.sample(self.max_degree, self.csn)):
            self.stubborn_supporters.append(nd2)
        for md2 in self.max_degree:
            if md2 not in self.stubborn_supporters:
                self.stubborn_opponents.append(md2)

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


def run_dynamics_limit_dissemination(information_layer, opinion_layer, alpha_a, frontier_extreme_number,
                                      central_extreme_number, omega_w, iteration_time, name, central_supporters_number,
                                     graph_number):
    G1 = information_layer
    G2 = opinion_layer
    alpha = alpha_a
    cen = central_extreme_number
    fen = frontier_extreme_number
    ow = omega_w
    mm = iteration_time
    csn = central_supporters_number
    source_list = list(nx.extrema_bounding(G1, compute='center'))

    path1 = 'E:/paper-data/parameter_setting/Limited information dissemination/w=' + str(ow) + '/graph' + \
            str(graph_number) + '/' + name
    for i in range(mm):
        folder = os.path.exists(path1 + '/number' + str(i))
        if not folder:
            os.makedirs(path1 + '/number' + str(i))
    source_node1 = source_list[0]
    ide = IoDynamics_limit_dissemination(G1, G2, source_node1, alpha, fen, cen, ow, csn)
    number_of_A = {}
    steady_opinion = {}
    for nodes in G2.nodes():
        number_of_A[nodes] = 0
        steady_opinion[nodes] = []

    for s1 in range(mm):
        ide2 = copy.deepcopy(ide)
        A_population, U_population, final_opinion, layer1_supporters_neighbors, layer2_supporters_neighbors, \
        layer1_opponents_neighbors, layer2_opponents_neighbors, source, nodes_A, first_ten_opinion = ide2.start1()
        b = 'E:/paper-data/parameter_setting/Limited information dissemination/w=' + str(ow) + '/graph' + \
            str(graph_number) + '/' + name + '/number' + str(s1) + '/'
        g1 = open(b + 'population of knowing news.txt', 'w')
        for i_1 in A_population:
            g1.write(str(i_1) + '\n')
        g1.close()

    f = open('E:/paper-data/parameter_setting/Limited information dissemination/w=' + str(ow) + '/graph' +
             str(graph_number) + '/' + name + '/probability of knowing news.txt', 'w')
    for keys1 in number_of_A:
        f.write(str(keys1) + ':' + str(number_of_A[keys1]) + '\n')
    f.close()
    g5 = open('E:/paper-data/parameter_setting/Limited information dissemination/w=' + str(ow) + '/graph' +
             str(graph_number) + '/' + name + '/average_opinion.txt', 'w')
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


if __name__ == '__main__':
    G1 = nx.read_edgelist('E:/paper-data/parameter_setting/initial opinion distribution/ba1')
    G2 = nx.read_edgelist('E:/paper-data/parameter_setting/initial opinion distribution/er2')
    N = nx.number_of_nodes(G1)
    Ns = 10
    m = 100
    t = 10
    alpha1 = 0.05

    distribution = ['random', 'power-law', 'normal']
    for type1 in distribution:
        for j in range(11):
              nnnn = '/' + str(j) + '-central extreme supporter'
              run_dynamics_opinion_distribution(G1, G2, alpha1, Ns, Ns, t, m, nnnn, j, type1)

    w = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for ow1 in w:
        for graph in range(20):
            path = 'E:/paper-data/parameter_setting/Limited information dissemination/network-matrix/'
            G1 = nx.read_edgelist(path + 'graph' + str(graph) + '/ba')
            G2 = nx.read_edgelist(path + 'graph' + str(graph) + '/er')
            for j in range(11):
                nnnn = '/' + str(j) + '-central extreme supporter'
                run_dynamics_limit_dissemination(G1, G2, alpha1, Ns, Ns, ow1, m, nnnn, j, graph)
