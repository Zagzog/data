import networkx as nx
import random
from main_program_support import degree_node
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from networkx.algorithms import community
import itertools
from pylab import *


def figure2_pre(graph_name, ces):
    name = graph_name + '/' + str(ces) + '-central extreme supporter/analysis results/'
    f = open('E:/paper-data/model data/' + name + '/simulation_frequency.txt', 'r')
    lines = f.readlines()
    f.close()
    opinions_sim = []
    for line in lines:
        item = line.strip('\n')
        opinions_sim.append(float(item))

    f = open('E:/paper-data/model data/' + name + '/theory_frequency.txt', 'r')
    lines = f.readlines()
    f.close()
    opinions_the = []
    for line in lines:
        item = line.strip('\n')
        opinions_the.append(float(item))

    opinion_x_sim = []
    for i1 in range(-10, 10, 1):
        opinion_x_sim.append(i1 / 10 + 0.05)
    opinion_x_the = []
    for i1 in range(-10, 10, 1):
        opinion_x_the.append(i1 / 10 + 0.05)
    return opinions_sim, opinions_the, opinion_x_sim, opinion_x_the


def figure2_plot(graph_name, ces, lab1):
    colors = ['#0000CC', '#0F0FB7', '#2424C2', '#3434C6', '#4747D5', '#5F66FD',
              '#6F6FD3', '#8888D3', '#9A9AC9', '#B0B0C9',
              '#D7BCBC', '#D19E9E', '#D28585', '#CD6C6C', '#FF6565',
              '#DE4848', '#DC3636', '#D12525', '#C31010', '#CC0000']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 60,
    }
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 40,
    }

    figure, ax = plt.subplots(figsize=(12, 7))
    x = [0]
    y = [0]
    plt.subplot(1, 2, 1)
    ba_er_sim, ba_er_the, x_sim, x_the = figure2_pre(graph_name + '/', ces)
    plt.plot(x_the, ba_er_the, 'o-', markersize=10, linewidth=3, label='theory', color='#FD7F0B')
    for xs in range(len(x_sim)):
        plt.bar(x_sim[xs], ba_er_sim[xs], width=0.09, align='center', facecolor=colors[xs])
    plt.bar(x, y, facecolor='white', edgecolor='gray', label='simulation')
    plt.xlim(-1.05, 1.05)
    plt.tick_params(labelsize=30)
    ax = plt.gca()
    ax.set_yscale('log')
    if graph_name == 'moscow':
        plt.ylim(1, 10000)
    elif graph_name == 'cannes':
        plt.ylim(1, 40000)
    else:
        plt.ylim(1, 12000)
    plt.yticks([1, 10, 100, 1000, 10000])
    plt.xticks([-1, -0.5, 0, 0.5, 1])
    # plt.title(lab1, font, x=-0.2)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    # plt.ylabel('frequency', font2, labelpad=13)
    # plt.xlabel('opinion', font2, labelpad=10)
    # if graph1 == 'BA-ER' and numb == 1:
    #     plt.legend(frameon=False, fontsize=20, loc='upper left', labelspacing=0.1)

    plt.subplot(1, 2, 2)
    plt.plot(x, y)
    plt.yticks([])
    plt.xticks([])
    ax = plt.gca()
    # plt.axis('off')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    plt.subplots_adjust(wspace=0, hspace=1, left=0.13, right=0.99, bottom=0.19, top=0.9)
    # plt.show()
    plt.savefig('F:/desktop/figure2/' + lab1 + '.png')


def figure3_plot(graph_name, ces, lab1):
    colors = ['#0000CC', '#0F0FB7', '#2424C2', '#3434C6', '#4747D5', '#5F66FD',
              '#6F6FD3', '#8888D3', '#9A9AC9', '#B0B0C9',
              '#D7BCBC', '#D19E9E', '#D28585', '#CD6C6C', '#FF6565',
              '#DE4848', '#DC3636', '#D12525', '#C31010', '#CC0000']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 60,
    }
    font2 = {
        'family': 'simhei',
        'weight': 'bold',
        'size': 40,
    }

    figure, ax = plt.subplots(figsize=(12, 7))
    x = [0]
    y = [0]
    plt.subplot(1, 2, 1)
    ba_er_sim, ba_er_the, x_sim, x_the = figure2_pre(graph_name + '/', ces)
    plt.plot(x_the, ba_er_the, 'o-', markersize=10, linewidth=3, label='theory', color='#FD7F0B')
    for xs in range(len(x_sim)):
        plt.bar(x_sim[xs], ba_er_sim[xs], width=0.09, align='center', facecolor=colors[xs])
    plt.bar(x, y, facecolor='white', edgecolor='gray', label='simulation')
    plt.xlim(-1.05, 1.05)
    plt.tick_params(labelsize=30)
    ax = plt.gca()
    ax.set_yscale('log')
    if graph_name == 'moscow':
        plt.ylim(1, 10000)
    elif graph_name == 'cannes':
        plt.ylim(1, 40000)
    else:
        plt.ylim(1, 12000)
    plt.yticks([1, 10, 100, 1000, 10000])
    plt.xticks([-1, -0.5, 0, 0.5, 1])
    # plt.title(lab1, font, x=-0.2)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    # plt.ylabel('frequency', font2, labelpad=13)
    # plt.xlabel('opinion', font2, labelpad=10)
    # if graph1 == 'BA-ER' and numb == 1:
    #     plt.legend(frameon=False, fontsize=20, loc='upper left', labelspacing=0.1)

    plt.subplot(1, 2, 2)
    plt.plot(x, y)
    plt.yticks([])
    plt.xticks([])
    ax = plt.gca()
    # plt.axis('off')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    plt.subplots_adjust(wspace=0, hspace=1, left=0.13, right=0.99, bottom=0.19, top=0.9)
    # plt.show()
    plt.savefig('F:/desktop/figure3/' + lab1 + '.png')


def figure4_plot(tci_name):
    color_list = ['#5582A8', '#FEC863', '#5EC4AA', '#EF5555', '#FF7B9B', '#B161F7']
    label_list = ['BA-BA', 'BA-ER', 'ER-BA', 'ER-ER', 'MOS', 'CAN']
    mpl.rcParams['axes.unicode_minus'] = False
    figure, ax = plt.subplots(figsize=(7, 5))
    font = {
        'weight': 'normal',
        'size': 30,
    }
    font2 = {
        'weight': 'normal',
        'size': 40,
    }
    graph_list = ['BA-BA', 'BA-ER', 'ER-BA', 'ER-ER', 'moscow', 'cannes']

    style_list = ['o-', 'D-', 's-', 'v-', '^-', '*-']

    for i in range(6):
        path = "F:/论文/SHUJU/model data/" + graph_list[i] + "/" + tci_name
        y = []
        f = open(path, "r")
        lines = f.readlines()
        f.close()
        for line in lines:
            item = line.strip('\n')
            y.append(float(item))

        x = [j for j in range(11)]

        plt.plot(x, y, style_list[i], markersize=8, color=color_list[i], label=label_list[i])

        # plt.legend(frameon=False, fontsize=15)
        # plt.ylim(-0.2, 1)
        plt.tick_params(labelsize=20)
        plt.subplots_adjust(wspace=0, hspace=1, left=0.2, right=0.9, bottom=0.15, top=0.9)
    # plt.show()
    plt.xticks([0, 2, 4, 6, 8, 10])
    plt.savefig('C:/Users/86178/Desktop/一审补充图/' + tci_name + ".png")


def figure5_plot(graph_name, graph_martix, numb):
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    font = {
        'weight': 'normal',
        'size': 30,
    }
    font2 = {
        'weight': 'normal',
        'size': 20,
    }
    G1 = graph_martix
    opinion_1 = [0 for i in range(21)]
    rr2 = []
    for i1 in range(-10, 11, 1):
        rr2.append(i1 / 10)
    rr = []
    for i1 in range(-10, 10, 1):
        rr.append((i1 / 10) + 0.05)
    max_min_dic = {}
    max_min_theta_list = []
    for i in range(51):
        max_min_dic[i] = []
        f = open('E:/paper-data/analytical approach/experimental-scheme/' + graph_name + '/result/'
                 + str(i) + '-frequency_proportion.txt', 'r')
        lines = f.readlines()
        f.close()
        z = []
        for line in lines:
            item = line.strip('\n')
            z.append(float(item))
        z1 = []
        for zz in range(len(z)):
            z1.append(z[zz] * rr[zz])

        avg_z = sum(z1) / sum(z)
        for i2 in range(21):
            if rr2[i2] <= avg_z <= rr2[i2 + 1]:
                pos = i2
                break
        max_min_dic[i].append(max(z))
        max_min_dic[i].append(rr[pos])
        # pos = z.index(max(z))
        # max_min_dic[i].append(rr[pos])

        min_degree, max_degree = degree_node(G1, i, 10+i)
        maxd = []
        for node in max_degree:
            dg = G1.degree(node)
            maxd.append(dg)
        mind = []
        for node in min_degree:
            dg = G1.degree(node)
            mind.append(dg)
        mad = sum(maxd) / len(maxd)
        if len(mind) == 0:
            mid = 0
        else:
            mid = sum(mind) / len(mind)
        max_min_theta = (i*mid + (i - 10) * mad) / ((10 + i) * mad + i*mid)
        max_min_theta_list.append(max_min_theta)
    x1 = [i*2 for i in range(51)]
    y1 = [max_min_dic[i][1] for i in range(51)]
    plt.scatter(x1, max_min_theta_list, marker='v', facecolors='none', edgecolors='#2B82B5', linewidth=1, s=60)
    plt.plot(x1, y1, color='#2B82B5')

    min_min_dic = {}
    min_min_theta_list = []
    for j in range(11):
        i = j*10
        min_min_dic[i] = []
        f = open('E:/paper-data/analytical approach/control-scheme/' + graph_name + '/result/'
                 + str(i) + '-frequency_proportion.txt', 'r')
        lines = f.readlines()
        f.close()
        z = []
        for line in lines:
            item = line.strip('\n')
            z.append(float(item))
        pos = z.index(max(z))
        min_min_dic[i].append(max(z))
        min_min_dic[i].append(rr[pos])
    for i in range(51):
        j = i*2
        min_degree, max_degree = degree_node(G1, j, 10)
        maxd = []
        for node in max_degree:
            dg = G1.degree(node)
            maxd.append(dg)
        mind = []
        for node in min_degree:
            dg = G1.degree(node)
            mind.append(dg)
        mad = sum(maxd) / len(maxd)
        if len(mind) == 0:
            mid = 0
        else:
            mid = sum(mind) / len(mind)
        min_min_theta = (j * mid - 10 * mad) / (10 * mad + j * mid)
        min_min_theta_list.append(min_min_theta)
    x2_the = [i*2 for i in range(51)]
    x2 = [i*10 for i in range(11)]
    y2 = [min_min_dic[i*10][1] for i in range(11)]
    plt.scatter(x2_the, min_min_theta_list, marker='o', facecolors='none', edgecolors='#F83719', linewidth=1, s=40)
    plt.plot(x2, y2, color='#F83719')

    plt.ylim(-1.1, 1.1)
    # plt.ylabel(''r'$\bar{\theta}$', font2, labelpad=10)
    # plt.xlabel('$L^+$', font2, labelpad=10)
    plt.tick_params(labelsize=18)
    # plt.title(numb, font, x=-0.2, y=1)
    plt.subplots_adjust(left=0.2, right=0.9, top=0.87, bottom=0.16)
    plt.savefig('F:/desktop/figure5/' + numb + '.png')
    plt.show()


def figure6_pre(graph_name, graph):
    all_c = []
    for i in range(11):
        name = str(graph_name) + '/' + str(i) + '-central extreme supporter'
        f = open('E:/paper-data/model data/' + name + '/simulation data/average_opinion.txt', 'r')
        lines = f.readlines()
        f.close()
        opinion = []
        for line in lines:
            item = line.strip('\n').split(':')
            opinion.append(float(item[1]))
        c = sum(opinion) / len(opinion)
        all_c.append(c)

    all_d = []
    G1 = graph
    min_degree, max_degree = degree_node(G1, 10, 10)
    maxd = []
    for node in max_degree:
        dg = G1.degree(node)
        maxd.append(dg)
    mind = []
    for node in min_degree:
        dg = G1.degree(node)
        mind.append(dg)
    mad = sum(maxd) / len(maxd)
    min = sum(mind) / len(mind)
    z = (mad - min) / (mad + min)
    # print(z)
    for x in range(11):
        d = (2 * float(x) - 10)/10 * z
        all_d.append(d)
    return all_c, all_d


def figure6_plot():
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    figure, ax = plt.subplots(figsize=(15, 7))
    colors = ['#60BB46', '#EE342B', '#3A53A4', '#AD59A3', '#5EC8DB', '#DAB061']
    ba_ba = nx.read_edgelist('E:/paper-data/network data/BA2_10000')
    er_ba = nx.read_edgelist('E:/paper-data/network data/BA1_10000')
    ba_er = nx.read_edgelist('E:/paper-data/network data/ER1_10000')
    er_er = nx.read_edgelist('E:/paper-data/network data/ER2_10000')
    mos = nx.read_edgelist('E:/paper-data/network data/moscow_3')
    can = nx.read_edgelist('E:/paper-data/network data/cannes_3')

    x = [i for i in range(11)]
    baba_1, baba_2 = figure6_pre('BA-BA', ba_ba)
    er_ba_1, er_ba_2 = figure6_pre('ER-BA', er_ba)
    ba_er1, ba_er2 = figure6_pre('BA-ER', ba_er)
    er_er_1, er_er_2 = figure6_pre('ER-ER', er_er)
    mos1, mos2 = figure6_pre('moscow', mos)
    can1, can2 = figure6_pre('cannes', can)

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'weight': 'bold',
        'size': 20,
    }
    font2 = {
        'weight': 'bold',
        'size': 30,
    }

    plt.subplot(1, 2, 1)
    plt.scatter(x, ba_er1, marker='v', facecolors='none', edgecolors=colors[0], linewidth=2, s=70, label='BA-ER')
    plt.plot(x, ba_er2, linestyle='--', linewidth=1, color=colors[0])
    plt.scatter(x, baba_1, marker='s', facecolors='none', edgecolors=colors[1], linewidth=2, s=70, label='BA-BA')
    plt.plot(x, baba_2, linestyle='--', linewidth=1, color=colors[1])
    plt.scatter(x, er_ba_1, marker='o', facecolors='none', edgecolors=colors[2], linewidth=2, s=70, label='ER-BA')
    plt.plot(x, er_ba_2, linestyle='--', linewidth=1, color=colors[2])
    plt.scatter(x, er_er_1, marker='p', facecolors='none', edgecolors=colors[3], linewidth=2, s=70, label='ER-ER')
    plt.plot(x, er_er_2, linestyle='--', linewidth=1, color=colors[3])

    plt.legend(frameon=False, fontsize=18)
    # plt.errorbar(x, ba_er1, error, color='w', fmt='.', ecolor='chocolate', capsize=5)
    plt.yticks([(10 - i*2)/10 for i in range(11)])
    plt.xticks([i for i in range(11)])
    # plt.ylabel(''r'$\bar{\theta}$', font)
    # plt.xlabel('x', font)
    plt.tick_params(labelsize=15)
    # plt.title('a', font2, x=-0.15, y=1)

    plt.subplot(1, 2, 2)
    plt.scatter(x, mos1, marker='^',  facecolors='none', edgecolors=colors[4], linewidth=2, s=70, label='MOS')
    plt.plot(x, mos2, linestyle='--', linewidth=1, color=colors[4])
    plt.scatter(x, can1, marker='D',  facecolors='none', edgecolors=colors[5], linewidth=2, s=70, label='CAN')
    plt.plot(x, can2, linestyle='--', linewidth=1, color=colors[5])

    plt.legend(frameon=False, fontsize=18)
    plt.yticks([(10 - i * 2) / 10 for i in range(11)])
    plt.xticks([i for i in range(11)])
    # plt.ylabel(''r'$\bar{\theta}$', font)
    # plt.xlabel('x', font)
    plt.tick_params(labelsize=15)
    # plt.title('b', font2, x=-0.15, y=1)

    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.07, right=0.99, bottom=0.15, top=0.85)
    plt.show()


# figure B.7
def spread_range_plot(graph_name, parm):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'weight': 'bold',
        'size': 60,
    }
    font2 = {
        'weight': 'bold',
        'size': 40,
    }

    figure, ax = plt.subplots(figsize=(10, 8))
    plt.tick_params(labelsize=30)
    h_list = [0.1, 1, 10, 100]
    color_list = ['#676767', '#42B497', '#D74B0C', '#3176B8']
    for i in range(4):
        if i == 1 and graph_name != "cannes" and graph_name != "moscow":
            path = "F:/论文/SHUJU/model data/" + graph_name + "/" + str(parm) + "-central extreme supporter/simulation data"
        else:
            path = "F:/paper-data/model-data-add/" + graph_name + "/h-" + str(h_list[i]) + "/" + str(parm) + \
                   "-central extreme supporter/simulation data"

        f = open(path + "/number_A.txt", "r")
        lines = f.readlines()
        f.close()

        y = []
        for line in lines:
            item = line.strip("\n")
            y.append(float(item))
        x = [_ for _ in range(len(y))]
        plt.plot(x, y, linewidth=3, color=color_list[i])
    plt.subplots_adjust(wspace=0, hspace=1, left=0.13, right=0.95, bottom=0.1, top=0.9)
    plt.show()


# figure B.8
def opinion_plot(graph_name, parm):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
        'weight': 'bold',
        'size': 60,
    }
    font2 = {
        'weight': 'bold',
        'size': 40,
    }

    figure, ax = plt.subplots(figsize=(8, 7))

    h_list = [0.1, 1, 10, 100]
    color_list = ['#676767', '#42B497', '#D74B0C', '#3176B8']  # 黑，绿，红，蓝
    for i in range(4):
        ba_er_sim, x_sim, = figure2_pre_ver2(graph_name + '/', h_list[i], parm)
        plt.plot(x_sim, ba_er_sim, 'o-', markersize=10, linewidth=4, color=color_list[i])
        avg_result = []
        for i in range(len(ba_er_sim)):
            avg_result.append(ba_er_sim[i] * x_sim[i])
        avg = sum(avg_result) / sum(ba_er_sim)
        plt.axvspan(-1.05, avg, facecolor='#FEFAEB')
        plt.axvspan(avg, 1.05, facecolor='#F2F7FC')
        ax = plt.gca()
        ax.set_yscale('symlog')

    plt.xlim(-1.05, 1.05)
    plt.tick_params(labelsize=30)
    plt.xticks([-1, -0.5, 0, 0.5, 1])
    x = [-1.05, 1.05]
    y = [0.65, 0.65]
    plt.plot(x, y, '--', linewidth=1, color='#5C5C5C')
    if graph_name == 'moscow':
        plt.ylim(1, 10000)
        plt.yticks([0.5, 1, 10, 100, 1000, 10000])
    elif graph_name == 'cannes':
        plt.ylim(1, 40000)
        plt.yticks([0.5, 1, 10, 100, 1000, 10000, 60000])
    else:
        plt.ylim(1, 12000)
        plt.yticks([0.5, 1, 10, 100, 1000, 10000])

    plt.show()


#  figure B.10-15
def SI_figure_frame(graph_name, lab1, ces):
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    colors = ['#0000CC', '#0F0FB7', '#2424C2', '#3434C6', '#4747D5', '#5F66FD',
              '#6F6FD3', '#8888D3', '#9A9AC9', '#B0B0C9',
              '#D7BCBC', '#D19E9E', '#D28585', '#CD6C6C', '#FF6565',
              '#DE4848', '#DC3636', '#D12525', '#C31010', '#CC0000']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {
                 'weight': 'bold',
                 'size': 60,
                 }
    font2 = {
                 'weight': 'normal',
                 'size': 40,
                 }

    figure, ax = plt.subplots(figsize=(12, 7))
    x = [0]
    y = [0]
    plt.subplot(1, 2, 1)
    ba_er_sim, ba_er_the, x_sim, x_the = figure2_pre(graph_name + '/', ces)
    plt.plot(x_the, ba_er_the, 'o-', markersize=10, linewidth=3, label='theory', color='#FD7F0B')
    for xs in range(len(x_sim)):
        plt.bar(x_sim[xs], ba_er_sim[xs], width=0.09, align='center', facecolor=colors[xs])
    plt.bar(x, y, facecolor='white', edgecolor='gray', label='simulation')
    plt.xlim(-1.05, 1.05)
    plt.tick_params(labelsize=30)
    ax = plt.gca()
    ax.set_yscale('log')
    if graph_name == 'moscow':
        plt.ylim(1, 10000)
    elif graph_name == 'cannes':
        plt.ylim(1, 40000)
    elif graph_name == 'CKM' or graph_name == 'CS':
        plt.ylim(1, 150)
    else:
        plt.ylim(1, 12000)
    if graph_name == 'CKM' or graph_name == 'CS':
        plt.yticks([1, 10, 100])
    else:
        plt.yticks([1, 10, 100, 1000, 10000])
    plt.xticks([-1, -0.5, 0, 0.5, 1])
    # plt.title(lab1, font, x=-0.2)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    # plt.ylabel('frequency', font2, labelpad=13)
    # plt.xlabel('opinion', font2, labelpad=10)
    # if graph1 == 'BA-ER' and numb == 1:
    #     plt.legend(frameon=False, fontsize=20, loc='upper left', labelspacing=0.1)

    plt.subplot(1, 2, 2)
    plt.plot(x, y)
    plt.yticks([])
    plt.xticks([])
    ax = plt.gca()
    # plt.axis('off')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    plt.subplots_adjust(wspace=0, hspace=1, left=0.11, right=0.97, bottom=0.19, top=0.9)
    # plt.savefig('F:/desktop/' + str(graph_name) + '/' + lab1[0] + '.png')
    plt.show()


