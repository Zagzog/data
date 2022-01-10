# -*- coding:utf-8 -*-

from numpy import *
import numpy as np
import networkx as nx
from pandas import DataFrame


def node_order(graph):
    G2 = graph
    ww = {}
    k = 0
    for node in G2.nodes():
        ww[node] = str(k)
        k += 1
    return ww


# calculate theory_probability_of_p
def theory_of_p(graph_name, graph_matrix):
    G1 = graph_matrix
    N = nx.number_of_nodes(G1)
    for i in range(11):
        path2 = 'E:/paper-data/model data/' + graph_name + '/' + str(i) + '-central extreme supporter'
        R = {}
        RR = {}
        source = []
        # find source
        f = open(path2 + '/simulation data/number0/source_node.txt', 'r')
        lines = f.readlines()
        f.close()
        for line1 in lines:
            source.append(line1)

        for i2 in G1.nodes():
            RR[str(i2)] = []
        for i1 in range(100):
            xx = {}
            r = {}
            f = open(path2 + '/simulation data/number' + str(i1) + '/opinion before silent.txt', 'r')
            lines = f.readlines()
            f.close()
            for item in lines:
                line = item.strip('\n').split(':')
                x = line[1]
                xx[line[0]] = []
                r[line[0]] = 0
                x2 = x.strip('[').strip(']').split(',')
                for i in range(len(x2)):
                    xx[line[0]].append(x2[i])
            for node in xx.keys():
                cc = []
                for j in range(len(xx[node])):
                    if xx[node][j] != '':
                        c = 0.1 / (1 + exp(-abs(float(xx[node][j]))))
                        cc.append(c)
                r[node] = sum(cc)
            for node in r.keys():
                RR[node].append(r[node])
        for key1 in RR.keys():
            R[key1] = sum(RR[key1]) / len(RR[key1])
        p_0 = {}
        P = {}
        p_store = {}
        p_store1 = {}
        for node in R.keys():
            if node not in source:
                p_0[node] = float(RR[node][0])
                P[node] = 0
                p_store[node] = float(RR[node][0])
                p_store1[node] = float(RR[node][0])
            else:
                p_0[node] = float(1)
                P[node] = 1
                p_store[node] = float(1)
                p_store1[node] = float(1)
        nn = 1
        while nn >= 0.001:
            nn = 0
            for node2 in G1.nodes():
                z = []
                y = 1
                for neighbor in G1.neighbors(node2):
                    xxx = 1 - R[neighbor] * p_store[neighbor]
                    z.append(xxx)
                for zz in z:
                    y = y * zz
                p_store1[node2] = 1 - y
            for key2 in p_store1.keys():
                if key2 not in source:
                    P[key2] = p_store[key2]
                    p_store[key2] = p_store1[key2]
                else:
                    P[key2] = 1
                    p_store[key2] = 1
            M = []
            for node in P:
                mm = abs(P[node] - p_store[node])
                M.append(mm)
            for item3 in M:
                if item3 >= 0.001:
                    nn = 1
                    break
        f = open(path2 + '/analysis results/theory_probability_of_p.txt', 'w')
        for key3, value3 in P.items():
            f.write(str(key3) + ':' + str(value3) + '\n')
        f.close()


# calculate theory_frequency
def theory_frequency(ces, graph_name, layer1, layer2):
    G1 = layer1
    G2 = layer2
    path = 'E:/paper-data/model data/' + str(graph_name) + '/' + str(ces) + '-central extreme supporter'
    f = open(path + '/analysis results/theory_probability_of_p.txt', 'r')
    lines = f.readlines()
    f.close()
    no_dic = node_order(G1)
    pp = {}
    for line in lines:
        item = line.strip('\n').split(':')
        pp[no_dic[item[0]]] = float(item[1])
    N2 = nx.number_of_nodes(G2)
    A = mat(zeros((N2, N2)))
    for edge1 in G2.edges():
        A[int(no_dic[edge1[0]]), int(no_dic[edge1[1]])] = pp[no_dic[edge1[0]]] * pp[no_dic[edge1[1]]]
        A[int(no_dic[edge1[1]]), int(no_dic[edge1[0]])] = pp[no_dic[edge1[0]]] * pp[no_dic[edge1[1]]]
    D = mat(zeros((N2, N2)))
    for i in range(N2):
        if A.sum(axis=1)[i] != 0:
            D[i, i] = A.sum(axis=1)[i]
        else:
            D[i, i] = 0.0000001

    P = np.linalg.inv(D) * A

    degrees = sum(list(D[i, i] for i in range(N2)))

    u = np.mat([1 for i in range(N2)]).T

    v = np.mat([D[i, i] / degrees for i in range(N2)])
    E = mat(eye(N2, N2))
    Q = (E-(P - u*v)).I

    g1 = open(path + '/simulation data/number0/opinion layer-extreme supporters neighbors.txt', 'r')
    lines = g1.readlines()
    g1.close()
    column1 = []
    column = []
    column2 = []
    for item1 in lines:
        line1 = item1.strip('\n').strip('[').strip(']').split(':')
        column1.append(int(no_dic[line1[0]]))
        column.append(int(no_dic[line1[0]]))

    g2 = open(path + '/simulation data/number0/opinion layer-extreme opponents neighbors.txt', 'r')
    lines = g2.readlines()
    g2.close()
    for item2 in lines:
        line2 = item2.strip('\n').strip('[').strip(']').split(':')
        column2.append(int(no_dic[line2[0]]))
        column.append(int(no_dic[line2[0]]))
    Q_1 = Q[column, :]
    Q11 = Q_1[:, column]
    raw1 = []
    for i in range(N2):
        if i not in column:
            raw1.append(i)
    Q_2 = Q[raw1, :]
    Q21 = Q_2[:, column]
    v1 = v[:, column]
    v2 = v[:, raw1]
    u1 = u[column, :]
    u2 = u[raw1, :]

    x = np.mat([0 for i in range(N2)]).T
    for value in column:
        if value in column1:
            x[value] = 1
        elif value in column2:
            x[value] = -1
    x1 = x[column, :]

    c = (v1*x1 + v2*Q21*np.linalg.inv(Q11)*x1) / (1 - (v2*u2 - v2*Q21*np.linalg.inv(Q11)*u1))

    x2 = c[0, 0]*u2 + Q21 * np.linalg.inv(Q11) * x1 - c[0, 0] * Q21 * np.linalg.inv(Q11) * u1

    opinion_1 = [0 for i in range(20)]
    rr = []
    for i1 in range(-10, 11, 1):
        rr.append(i1 / 10)
    if graph_name == 'CKM' or graph_name == 'CS':
        for k in range(N2 - 2):
            for i2 in range(20):
                if rr[i2] <= x2[k, 0] <= rr[i2+1]:
                    opinion_1[i2] += 1
                    break
        opinion_1[0] += 1
        opinion_1[-1] += 1
    else:
        for k in range(N2 - 20):
            for i2 in range(20):
                if rr[i2] <= x2[k, 0] <= rr[i2+1]:
                    opinion_1[i2] += 1
                    break
        opinion_1[0] += 10
        opinion_1[-1] += 10
    g = open(path + '/analysis results/theory_frequency.txt', 'w')
    for opinion in opinion_1:
        g.write(str(opinion) + '\n')
    g.close()


# TCI calculation method
def TCI(graph, opinion_dic):
    neighbor_relationship = {}
    degree_nodes = {}
    node_list = []

    for node in opinion_dic.keys():
        neighbor_relationship[node] = []
        node_list.append(node)
    # store neighbor relationship
    M = 0
    for node in node_list:
        mm = 0
        for neighbor in graph.neighbors(node):
            if neighbor in node_list:
                neighbor_relationship[node].append(neighbor)
                M += 1
                mm += 1
        degree_nodes[node] = mm
    product_opinion1 = []
    product_opinion2 = []
    for node1 in node_list:
        store_product_opinion1 = []
        store_product_opinion2 = []
        for node2 in node_list:
            average_degree_double = degree_nodes[node1] * degree_nodes[node2] / M
            if node2 in neighbor_relationship[node1]:
                product_factor = 1 - average_degree_double
            else:
                product_factor = 0 - average_degree_double
            if node2 == node1:
                product_factor2 = degree_nodes[node1] - average_degree_double
            else:
                product_factor2 = 0 - average_degree_double
            opinion_double1 = product_factor * opinion_dic[node1] * opinion_dic[node2]
            opinion_double2 = product_factor2 * opinion_dic[node1] * opinion_dic[node2]
            # degree_double1 = product_factor * degree_nodes[node1] * degree_nodes[node2]
            # degree_double2 = product_factor2 * degree_nodes[node1] * degree_nodes[node2]
            store_product_opinion1.append(opinion_double1)
            store_product_opinion2.append(opinion_double2)
            # store_product_degree1.append(degree_double1)
            # store_product_degree2.append(degree_double2)
        product_opinion1.append(sum(store_product_opinion1))
        product_opinion2.append(sum(store_product_opinion2))
        # product_degree1.append(sum(store_product_degree1))
        # product_degree2.append(sum(store_product_degree2))
    z = []
    zz = []
    for ii in product_opinion1:
        if ii > 0:
            z.append(ii)
        else:
            zz.append(ii)
    r = sum(product_opinion1) / sum(product_opinion2)
    # r1 = sum(product_degree1) / sum(product_degree2)
    return r


# calculate TCI
def calculate_TCI(graph_name, graph_matrix):
    G1 = graph_matrix
    tci_list = []
    for i in range(11):
        path = graph_name + '/' + str(i) + '-central extreme supporter/'
        opinion_dic = {}
        opinion_list = []

        f = open(path + 'simulation data/average_opinion.txt', 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            line1 = line.strip('\n').split(':')
            opinion_dic[line1[0]] = float(line1[1])
            opinion_list.append(float(line1[1]))
        original_r = TCI(G1, opinion_dic)
        tci_list.append(original_r)
    f = open('E:/paper-data/model data/' + graph_name + '/TCI.txt', 'w')
    for item in tci_list:
        f.write(str(item) + '\n')
    f.close()


# calculate simulation_frequency
def simulation_frequency(graph_name, ces):
    opinion_1 = [0 for i in range(20)]
    rr = []
    for i1 in range(-10, 11, 1):
        rr.append(i1 / 10)
    path = 'E:/paper-data/model data/' + str(graph_name) + '/' + str(ces) + '-central extreme supporter'
    f = open(path + '/simulation data/average_opinion.txt', 'r')
    lines1 = f.readlines()
    f.close()
    for item in lines1:
        line1 = item.strip('\n').split(':')
        for i2 in range(20):
            if rr[i2] <= float(line1[1]) <= rr[i2 + 1]:
                opinion_1[i2] += 1
                break
    g = open(path + '/analysis results/simulation_frequency.txt', 'w')
    for opinion in opinion_1:
        g.write(str(opinion) + '\n')
    g.close()


# calculate average result of population
def limit_of_omega():
    path = 'E:/paper-data/parameter_setting/Limited information dissemination/'
    folder = os.path.exists(path + 'average result of population')
    if not folder:
        os.makedirs(path + 'average result of population')
    for k in range(11):
        name = '/' + str(k) + '-central extreme supporter/'
        all_number = []
        for z in range(1, 10):
            number_2 = []
            for j in range(20):
                number = []
                for i in range(100):
                    path1 = 'w=' + str(z) + '/graph' + str(j) + name + '/number' + str(i)
                    f = open(path + path1 + '/population of knowing news.txt', 'r')
                    lines = f.readlines()
                    f.close()
                    num = []
                    for item in lines:
                        line = item.strip('\n')
                        num.append(float(line))
                    number.append(num[-1])
                number_2.append(sum(number) / 100)
            all_number.append(sum(number_2) / 20)
        f = open(path + 'average result of population' + name + '.txt', 'w')
        for nn in all_number:
            f.write(str(nn) + '\n')
        f.close()


# network view
def edge_select(ces, graph_matrix, graph_name):
    path1 = 'E:/paper-data/network data/'
    G = nx.read_edgelist(path1 + graph_matrix)
    no_1 = node_order(G)

    path = 'E:/paper-data/model data/' + graph_name + '/' + str(ces) + '-central extreme supporter/simulation data/'
    f = open(path + 'average_opinion.txt', 'r')
    lines = f.readlines()
    opinion = {}
    for line in lines:
        item = line.strip('\n').split(':')
        opinion[item[0]] = item[1]
    data1 = {
        'node': [],
        'type': [],
        'degree': []
    }
    data = {
        'edge1': [],
        'edge2': []
    }
    node_store = []

    path1 = 'E:/paper-data/model data/' + graph_name + '/1-central extreme supporter/simulation data/'
    f = open(path1 + 'number0/opinion layer-extreme supporters neighbors.txt', 'r')
    lines = f.readlines()
    for line in lines:
        item = line.strip('\n').split(':')
        node_store.append(item[0])
        item1 = item[1].strip('[').strip(']').split(',')
        for value in item1:
            node1 = value.strip(' ').strip("'")
            node_store.append(node1)

    f = open(path1 + 'number0/opinion layer-extreme opponents neighbors.txt', 'r')
    lines = f.readlines()
    for line in lines:
        item = line.strip('\n').split(':')
        node_store.append(item[0])
        item1 = item[1].strip('[').strip(']').split(',')
        for value in item1:
            node1 = value.strip(' ').strip("'")
            node_store.append(node1)

    path2 = 'E:/paper-data/model data/' + graph_name + '/5-central extreme supporter/simulation data/'
    f = open(path2 + 'number0/opinion layer-extreme supporters neighbors.txt', 'r')
    lines = f.readlines()
    for line in lines:
        item = line.strip('\n').split(':')
        node_store.append(item[0])
        item1 = item[1].strip('[').strip(']').split(',')
        for value in item1:
            node1 = value.strip(' ').strip("'")
            node_store.append(node1)

    f = open(path2 + 'number0/opinion layer-extreme opponents neighbors.txt', 'r')
    lines = f.readlines()
    for line in lines:
        item = line.strip('\n').split(':')
        node_store.append(item[0])
        item1 = item[1].strip('[').strip(']').split(',')
        for value in item1:
            node1 = value.strip(' ').strip("'")
            node_store.append(node1)

    path3 = 'E:/paper-data/model data/' + graph_name + '/9-central extreme supporter/simulation data/'
    f = open(path3 + 'number0/opinion layer-extreme supporters neighbors.txt', 'r')
    lines = f.readlines()
    for line in lines:
        item = line.strip('\n').split(':')
        node_store.append(item[0])
        item1 = item[1].strip('[').strip(']').split(',')
        for value in item1:
            node1 = value.strip(' ').strip("'")
            node_store.append(node1)

    f = open(path3 + 'number0/opinion layer-extreme opponents neighbors.txt', 'r')
    lines = f.readlines()
    for line in lines:
        item = line.strip('\n').split(':')
        node_store.append(item[0])
        item1 = item[1].strip('[').strip(']').split(',')
        for value in item1:
            node1 = value.strip(' ').strip("'")
            node_store.append(node1)

    if graph_name != 'BA-ER' and graph_name != 'ER-ER':
        node_choose = list(set(node_store))
        node_choose2 = []
        for node in node_choose:
            if node in opinion.keys():
                data1['node'].append(node)
                data1['type'].append(opinion[node])
                data1['degree'].append(G.degree(node))
                node_choose2.append(node)
        edge_select = []
        for edge1 in G.edges():
            if edge1[0] in node_choose2 and edge1[1] in node_choose2:
                edge_select.append(edge1)
        edge_choose = list(set(edge_select))
        for edge in edge_choose:
            data['edge1'].append(edge[0])
            data['edge2'].append(edge[1])
    else:
        node_store2 = []
        for node in node_store:
            node_store2.append(node)
            for neighbors in G.neighbors(node):
                node_store2.append(neighbors)
        node_choose = list(set(node_store2))
        node_choose2 = []
        for node in node_choose:
            if node in opinion.keys():
                data1['node'].append(node)
                data1['type'].append(opinion[node])
                data1['degree'].append(G.degree(node))
                node_choose2.append(node)
        edge_select = []
        for edge1 in G.edges():
            if edge1[0] in node_choose2 and edge1[1] in node_choose2:
                edge_select.append(edge1)
        edge_choose = list(set(edge_select))
        for edge in edge_choose:
            data['edge1'].append(edge[0])
            data['edge2'].append(edge[1])
    df = DataFrame(data)
    df.to_excel('E:/paper-data/network_view/' + graph_name + '/' + graph_name + '-edges.xlsx')
    df1 = DataFrame(data1)
    df1.to_excel('E:/paper-data/network_view/' + graph_name + '/' + graph_name + str(ces) + '.xlsx')


if __name__ == '__main__':
    c = [i for i in range(11)]
    graph_list = ['BA-ER', 'BA-BA', 'ER-BA', 'ER-ER', 'moscow', 'cannes']
    graph_matrix = ['ER1_10000', 'BA2_10000', 'BA1_10000', 'ER2_10000', 'moscow_3', 'cannes_3']
    for graph in range(len(graph_list)):
        for i in c:
            edge_select(i, graph_matrix[graph], graph_list[graph])
