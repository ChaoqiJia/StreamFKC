import numpy as np
import random
import itertools
import csv
import math
from copy import deepcopy
from scipy.spatial import distance
from collections import Counter


########################################################################################################################

def min_metric(x, X, metric='euclidean'):
    # distance_matrix = distance.cdist(x, X, metric).flatten()
    return np.min(distance.cdist(x, X, metric))


########################################################################################################################
########################################################################################################################
def fiveApprox(constraints, epsilon, PATH):
    # streaming

    para = 1 + epsilon
    left = 0
    right = 0
    m = len(constraints)
    Gamma_init_X = [-1] * m
    R_dict = dict()
    save_center = [[] for _ in range(m)]
    j = -1
    csv_filename = PATH

    with open(csv_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        i = -1

        for row in csv_reader:
            Y = int(row[-1])
            X = [float(item) for item in row[:-1]]

            if Y != i:
                Gamma_init_X[Y] = X
                i = Y

            if len(save_center[Y]) < 2 * constraints[Y]:
                save_center[Y].append(X)

            left, right, j = OurAlg1(Gamma_init_X, X, Y, constraints, m, para, R_dict, left, right, j, metric='euclidean')

    filtered_dict = {k: v for k, v in R_dict.items() if left < k <= right}


    # postprocessing
    right = float('inf')
    left = 0
    C = []
    for fd, fd_value in filtered_dict.items():
        if left < fd < right:
            fd_lengths = np.array([len(value) for value in fd_value])
            is_valid = np.all(fd_lengths <= constraints)
            if is_valid:
                C = [item for item in (fd_value[0] + fd_value[1])]
                for num in range(2):
                    i = 0
                    for save_center_item in save_center[num]:
                        if save_center_item not in C:
                            if i >= constraints[num] - len(fd_value[num]):
                                break
                            C.append(save_center_item)
                            i += 1
                right = fd
            else:
                C_copy = OurAlg3(fd_value, constraints, fd, save_center, metric='euclidean')
                if C_copy == -1:
                    left = fd
                else:
                    C = C_copy
                    right = fd
    print(len(C))
    return C


#######################################################################################################################
# constructure the independent set for k groups_ fixed the real value needed5423
def OurAlg1(Gamma_init_X, X, classTable, constraints, m, para, R_dict, left, right, j_bound, metric='euclidean'):
    # each points
    Gamma_x = X  # currently point
    j = j_bound + 1
    right = max(right, np.linalg.norm(np.array(Gamma_x) - np.array(Gamma_init_X[classTable])))

    while left < (para - 1) * (para ** j) <= right:
        R = round((para - 1) * (para ** j), 2)
        if R not in R_dict:
            R_dict[R] = [[] for _ in range(m)]
        Gamma_para = R_dict[R]
        if len(Gamma_para[classTable]) == 0:
            Gamma_para[classTable].append(Gamma_init_X[classTable])

        second_elements = Gamma_para[classTable]
        if min_metric([Gamma_x], second_elements, metric) > R * 2:
            Gamma_para[classTable].append(Gamma_x)

        if max(len(Gamma_para[0]), len(Gamma_para[1])) > constraints[0] + constraints[1]:
            left = R
            j_bound = j
        j += 1
    # return left, right, j_bound
    return left, right, j_bound


########################################################################################################################

########################################################################################################################
# this version not sure which part of the points can be used
def OurAlg3(fd_value, constraints, fd, save_center, metric='euclidean'):
    Graph = {tuple(key): [] for key in fd_value[0] + fd_value[1]}

    # Glg.2
    for l_node, r_node in itertools.product(fd_value[0], fd_value[1]):
        if np.linalg.norm(np.array(l_node) - np.array(r_node)) <= fd * 3:
            Graph[tuple(l_node)].append(tuple(r_node))
            Graph[tuple(r_node)].append(tuple(l_node))

    C = [np.array(key) for key, value in Graph.items() if len(value) == 0]
    Graph = {key: value for key, value in Graph.items() if not any(np.array_equal(key, c) for c in C)}

    while True:
        C_count = [np.intersect1d(C, fd_value[0]), np.intersect1d(C, fd_value[1])]
        Graph_count = [np.intersect1d(np.array(list(Graph.keys())), fd_value[0]), np.intersect1d(np.array(list(Graph.keys())), fd_value[1])]
        if len(C_count[0]) > constraints[0] or len(C_count[1]) > constraints[1]:
            return -1


        if len(C_count[0]) + len(Graph_count[0]) <= constraints[0]:
            C = [*C, *fd_value[0]]
            break
        elif len(C_count[1]) + len(Graph_count[1]) <= constraints[1]:
            C = [*C, *fd_value[1]]
            break

        one_degree_nodes = [key for key, value in Graph.items() if len(value) == 1]
        neighbors = []

        if len(one_degree_nodes) != 0:
            for node in one_degree_nodes:
                neighbors.append(Graph[node][0])

            most_common_element = Counter(neighbors).most_common(1)[0][0]
            C.append(most_common_element)

            for node in Graph[most_common_element]:
                Graph[node].remove(most_common_element)
                if len(Graph[node]) == 0:
                    del Graph[node]
            del Graph[most_common_element]
            # Graph.remove[Graph[most_common_element]]

        else:
            p = random.choice(list(Graph.keys()))
            C.append(p)
            q = random.choice(Graph[p])
            for p_n in Graph[p]:
                Graph[p_n].remove(p)

            for q_n in Graph[q]:
                Graph[q_n].remove(q)

            del Graph[p]
            del Graph[q]
    # print(C)
    C_count = [[x for x in C if tuple(x) in map(tuple, fd_value[0])], [x for x in C if tuple(x) in map(tuple, fd_value[1])]]
    if len(C_count[0]) > constraints[0] or len(C_count[1]) > constraints[1]:
        return -1
    for num in range(2):
        i = 0
        for save_center_item in save_center[num]:
            if save_center_item not in C:
                if i >= constraints[num] - len(C_count[num]):
                    break
                C.append(save_center_item)
                i += 1
    print(len(C))
    return C
