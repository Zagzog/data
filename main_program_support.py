# -*- coding:utf-8 -*-

import networkx as nx
import random
import numpy as np


# select extreme nodes
def degree_node(graph, number1, number2):
    # p is small degree
    # q is large degree
    p = []
    q = []
    d = dict(nx.degree(graph))
    b = nx.degree_histogram(graph)
    b1 = {}
    for long in range(len(b)):
        b1[long] = []
    for node, degrees in d.items():
        b1[degrees].append(node)
    c = 0
    for i in range(len(b)):
        if b[i] + c >= number1:
            m = number1 - c
            k = i
            break
        else:
            c = b[i] + c
    for j in range(k):
        if b[j] != 0:
            for y in b1[j]:
                p.append(y)
    for node1 in random.sample(b1[k], m):
        p.append(node1)

    h = 0
    for i1 in range(len(b)-1, -1, -1):
        if b[i1] + h >= number2:
            m1 = number2 - h
            k1 = i1
            break
        else:
            h = b[i1] + h

    for j1 in range(len(b)-1, k1, -1):
        if b[j1] != 0:
            #print(b1[j1])
            for y1 in b1[j1]:
                q.append(y1)
    for node2 in random.sample(b1[k1], m1):
        q.append(node2)
    return p, q


# power-law distribution
def opinion_start2(nodes1):
    r = []
    opinion = {}
    for i in range(-100, 101, 10):
        r.append((i / 100)*1.0)
    p = []
    for oo in range(1, 21):
        p.append((oo ** -2)*1.0)
    r1 = []
    for k in p:
        r1.append((k*1.0) / (sum(p))*1.0)
    p1 = np.array(r1)
    for node in nodes1:
        pp = np.random.choice(list(range(20)), p=p1.ravel())
        rr = random.uniform(r[pp], r[pp+1])
        opinion[node] = rr
    return opinion


# random distribution
def opinion_start3(nodes1):
    opinion = {}
    for i in nodes1:
        r = random.uniform(-1, 1)
        opinion[i] = r
    return opinion


# normal distribution
def opinion_start4(nodes1, avg=1.5, dev=0.1):
    opinion = {}
    gaussion = []
    opinion_store = {}
    c = []
    for k in range(100):
        c.append(random.random())
    for kk in range(20):
        opinion_store[kk] = [-1 + kk/10, -0.9 + kk/10]
    for i in range(21):
        z = round(np.exp(-0.5 * (i / 10 - avg)**2 / dev) * (1 / (2 * np.pi * dev)**0.5), 4)
        gaussion.append(z)
    gaussion1 = []
    gg1 = 0
    for gg in gaussion:
        gg1 = gg + gg1
        gaussion1.append(gg1)
    for node in nodes1:
        r = random.choice(c)
        if r <= gaussion1[0]:
            opinion[node] = -1
        elif r >= gaussion1[-1]:
            opinion[node] = 1
        else:
            for j in range(20):
                if gaussion1[j] <= r < gaussion1[j+1]:
                    opinion[node] = random.uniform(opinion_store[j][0], opinion_store[j][1])
                    break
    return opinion


# generate synthetic networks
def graph(n):
    G = nx.Graph
    N = n
    H = nx.Graph
    B = nx.barabasi_albert_graph(N, 5)
    A = nx.barabasi_albert_graph(N, 5)
    K = False
    while not K:
        G = nx.erdos_renyi_graph(N, 0.002)
        K = nx.is_connected(G)
    K1 = False
    while not K1:
        H = nx.erdos_renyi_graph(N, 0.002)
        K1 = nx.is_connected(H)
    return A, B, G, H


def CKM_graph():
    path = 'E:/paper-data/network data/CKM-Physicians-Innovation_Multiplex_Social/Dataset/'
    f = open(path + 'CKM-Physicians-Innovation_multiplex.edges', 'r')
    lines = f.readlines()
    f.close()
    layer1 = {}
    layer2 = {}
    layer3 = {}
    for item in lines:
        line = item.strip('\n').split(' ')
        layer1[line[1]] = []
        layer2[line[1]] = []
        layer3[line[1]] = []
    for item in lines:
        line = item.strip('\n').split(' ')
        if line[0] == '1':
            layer1[line[1]].append(line[2])
        elif line[0] == '2':
            layer2[line[1]].append(line[2])
        elif line[0] == '3':
            layer3[line[1]].append(line[2])
    G1 = nx.Graph()
    G2 = nx.Graph()
    G3 = nx.Graph()
    for k1, v1 in layer1.items():
        if v1:
            for v11 in v1:
                G1.add_edge(k1, v11)
    for k2, v2 in layer2.items():
        if v2:
            for v22 in v2:
                G2.add_edge(k2, v22)
    for k3, v3 in layer3.items():
        if v3:
            for v33 in v3:
                G3.add_edge(k3, v33)
    nodes_of_G2 = [node for node in G2.nodes()]
    nodes_of_G3 = [node for node in G3.nodes()]
    nodes = []
    for node in nodes_of_G2:
        if node in nodes_of_G3:
            nodes.append(node)
    for node in nodes_of_G3:
        if node in nodes_of_G2:
            if node not in nodes:
                nodes.append(node)
    for node in nodes_of_G2:
        if node not in nodes:
            G2.remove_node(node)
    for node in nodes_of_G3:
        if node not in nodes:
            G3.remove_node(node)
    K = False
    J = False
    while not K or not J:
        H2 = max(nx.connected_component_subgraphs(G2), key=len)
        H3 = max(nx.connected_component_subgraphs(G3), key=len)
        m = []
        n = []
        for node in H2.nodes():
            if node not in H3.nodes():
                m.append(node)
        for node in H2.nodes():
            if node not in H3.nodes():
                n.append(node)
        for m1 in m:
            H2.remove_node(m1)
        for n1 in n:
            H3.remove_node(n1)
        K = nx.is_connected(H2)
        J = nx.is_connected(H3)
        G2 = H2
        G3 = H3
    nx.write_edgelist(G2, 'E:/paper-data/network data/CKM_layer2')
    nx.write_edgelist(G3, 'E:/paper-data/network data/CKM_layer3')
    return G2, G3


def CS_graph():
    path = 'E:/paper-data/network data/CS-Aarhus_Multiplex_Social/Dataset/'
    f = open(path + 'CS-Aarhus_multiplex.edges', 'r')
    lines = f.readlines()
    f.close()
    layer1, layer2, layer3, layer4, layer5 = {}, {}, {}, {}, {}
    for item in lines:
        line = item.strip('\n').split(' ')
        layer1[line[1]] = []
        layer2[line[1]] = []
        layer3[line[1]] = []
        layer4[line[1]] = []
        layer5[line[1]] = []
    for item in lines:
        line = item.strip('\n').split(' ')
        if line[0] == '1':
            layer1[line[1]].append(line[2])
        elif line[0] == '2':
            layer2[line[1]].append(line[2])
        elif line[0] == '3':
            layer3[line[1]].append(line[2])
        elif line[0] == '4':
            layer4[line[1]].append(line[2])
        elif line[0] == '5':
            layer5[line[1]].append(line[2])
    G1 = nx.Graph()
    G2 = nx.Graph()
    G3 = nx.Graph()
    G4 = nx.Graph()
    G5 = nx.Graph()
    for k1, v1 in layer1.items():
        if v1:
            for v11 in v1:
                G1.add_edge(k1, v11)
    for k2, v2 in layer2.items():
        if v2:
            for v22 in v2:
                G2.add_edge(k2, v22)
    for k3, v3 in layer3.items():
        if v3:
            for v33 in v3:
                G3.add_edge(k3, v33)
    for k4, v4 in layer4.items():
        if v4:
            for v44 in v4:
                G4.add_edge(k4, v44)
    for k5, v5 in layer5.items():
        if v5:
            for v55 in v5:
                G5.add_edge(k5, v55)
    nodes_of_G2 = [node for node in G2.nodes()]
    nodes_of_G5 = [node for node in G5.nodes()]
    nodes = []
    for node in nodes_of_G2:
        if node in nodes_of_G5:
            nodes.append(node)
    for node in nodes_of_G5:
        if node in nodes_of_G2:
            if node not in nodes:
                nodes.append(node)
    for node in nodes_of_G2:
        if node not in nodes:
            G2.remove_node(node)
    for node in nodes_of_G5:
        if node not in nodes:
            G5.remove_node(node)
    nx.write_edgelist(G2, 'E:/paper-data/network data/CS-Aarhus_layer2')
    nx.write_edgelist(G5, 'E:/paper-data/network data/CS-Aarhus_layer5')
    return G2, G5


def cannes_graph():
    path = 'E:/paper-data/network data/Cannes2013_Multiplex_Social/Dataset/'
    f = open(path + 'Cannes2013_multiplex.edges', 'r')
    lines = f.readlines()
    f.close()
    layer1 = {}
    layer2 = {}
    layer3 = {}
    for item in lines:
        line = item.strip('\n').split(' ')
        layer1[line[1]] = []
        layer2[line[1]] = []
        layer3[line[1]] = []
    for item in lines:
        line = item.strip('\n').split(' ')
        if line[0] == '1':
            layer1[line[1]].append(line[2])
        elif line[0] == '2':
            layer2[line[1]].append(line[2])
        elif line[0] == '3':
            layer3[line[1]].append(line[2])
    G1 = nx.Graph()
    G2 = nx.Graph()
    G3 = nx.Graph()
    for k1, v1 in layer1.items():
        if v1:
            for v11 in v1:
                G1.add_edge(k1, v11)
    for k2, v2 in layer2.items():
        if v2:
            for v22 in v2:
                G2.add_edge(k2, v22)
    for k3, v3 in layer3.items():
        if v3:
            for v33 in v3:
                G3.add_edge(k3, v33)
    nodes_of_G2 = [node for node in G2.nodes()]
    nodes_of_G3 = [node for node in G3.nodes()]
    nodes = []
    for node in nodes_of_G2:
        if node in nodes_of_G3:
            nodes.append(node)
    for node in nodes_of_G3:
        if node in nodes_of_G2:
            if node not in nodes:
                nodes.append(node)
    for node in nodes_of_G2:
        if node not in nodes:
            G2.remove_node(node)
    for node in nodes_of_G3:
        if node not in nodes:
            G3.remove_node(node)
    K = False
    J = False
    while not K or not J:
        H2 = max(nx.connected_component_subgraphs(G2), key=len)
        H3 = max(nx.connected_component_subgraphs(G3), key=len)
        m = []
        n = []
        for node in H2.nodes():
            if node not in H3.nodes():
                m.append(node)
        for node in H3.nodes():
            if node not in H2.nodes():
                n.append(node)
        for m1 in m:
            H2.remove_node(m1)
        for n1 in n:
            H3.remove_node(n1)
        K = nx.is_connected(H2)
        J = nx.is_connected(H3)
        G2 = H2
        G3 = H3
    nx.write_edgelist(G2, 'E:/paper-data/network data/cannes_2')
    nx.write_edgelist(G3, 'E:/paper-data/network data/cannes_3')
    return G2, G3


def moscow_graph():
    path = 'E:/paper-data/network data/MoscowAthletics2013_Multiplex_Social/Dataset/'
    f = open(path + 'MoscowAthletics2013_multiplex.edges', 'r')
    lines = f.readlines()
    f.close()
    layer1 = {}
    layer2 = {}
    layer3 = {}
    for item in lines:
        line = item.strip('\n').split(' ')
        layer1[line[1]] = []
        layer2[line[1]] = []
        layer3[line[1]] = []
    for item in lines:
        line = item.strip('\n').split(' ')
        if line[0] == '1':
            layer1[line[1]].append(line[2])
        elif line[0] == '2':
            layer2[line[1]].append(line[2])
        elif line[0] == '3':
            layer3[line[1]].append(line[2])
    G1 = nx.Graph()
    G2 = nx.Graph()
    G3 = nx.Graph()
    for k1, v1 in layer1.items():
        if v1:
            for v11 in v1:
                G1.add_edge(k1, v11)
    for k2, v2 in layer2.items():
        if v2:
            for v22 in v2:
                G2.add_edge(k2, v22)
    for k3, v3 in layer3.items():
        if v3:
            for v33 in v3:
                G3.add_edge(k3, v33)
    nodes_of_G2 = [node for node in G2.nodes()]
    nodes_of_G3 = [node for node in G3.nodes()]
    nodes = []
    for node in nodes_of_G2:
        if node in nodes_of_G3:
            nodes.append(node)
    for node in nodes_of_G3:
        if node in nodes_of_G2:
            if node not in nodes:
                nodes.append(node)
    for node in nodes_of_G2:
        if node not in nodes:
            G2.remove_node(node)
    for node in nodes_of_G3:
        if node not in nodes:
            G3.remove_node(node)
    K = False
    J = False
    while not K or not J:
        H2 = max(nx.connected_component_subgraphs(G2), key=len)
        H3 = max(nx.connected_component_subgraphs(G3), key=len)
        m = []
        n = []
        for node in H2.nodes():
            if node not in H3.nodes():
                m.append(node)
        for node in H3.nodes():
            if node not in H2.nodes():
                n.append(node)
        for m1 in m:
            H2.remove_node(m1)
        for n1 in n:
            H3.remove_node(n1)
        K = nx.is_connected(H2)
        J = nx.is_connected(H3)
        G2 = H2
        G3 = H3
    nx.write_edgelist(G2, 'E:/paper-data/network data/moscow_2')
    nx.write_edgelist(G3, 'E:/paper-data/network data/moscow_3')
    return G2, G3


if __name__ == '__main__':
    A, B, C, D = graph(10000)
    nx.write_edgelist(A, 'E:/paper-data/network data/BA1_10000')
    nx.write_edgelist(B, 'E:/paper-data/network data/BA2_10000')
    nx.write_edgelist(C, 'E:/paper-data/network data/ER1_10000')
    nx.write_edgelist(D, 'E:/paper-data/network data/ER2_10000')
