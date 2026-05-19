import numpy as np
import csv
from scipy.spatial import distance


########################################################################################################################

# def min_metric(x, X, metric='euclidean'):
#     if not X:
#         distance_min = sys.maxsize
#     else:
#         distance_matrix = distance.cdist(x, X, metric).flatten()
#         distance_min = np.min(distance_matrix)
# return distance_min
def min_metric(x, X, metric='euclidean'):
    if len(X) == 1:
        return np.linalg.norm(np.array(x) - X)
    return np.min(distance.cdist(x, X, metric))



########################################################################################################################
########################################################################################################################
def fourApprox(constraints, epsilon, PATH, PATH1 = 'output.txt'):
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
    # group_file = open(PATH1, 'r')

    with open(csv_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        i = -1

        for row in csv_reader:
            # Y = int(group_file.readline().strip())
            Y = int(row[-1])
            X = [float(item) for item in row[:-1]]
            if Y != i:
                Gamma_init_X[Y] = X
                i = Y

            if len(save_center[Y]) < 2 * constraints[Y]:
                save_center[Y].append(X)

            left, right, j = ICS(Gamma_init_X, X, Y, constraints, m, para, R_dict, left, right, j, metric='euclidean')

    filtered_dict = {k: v for k, v in R_dict.items() if left < k <= right}

    # postprocessing
    right = float('inf')
    left = 0
    C = []
    for fd, fd_value in filtered_dict.items():
        if left < fd < right:
            if len(fd_value[0]) <= constraints[0]:
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
            elif len(fd_value[0]) + len(fd_value[1]) <= constraints[0] + constraints[1]:
                left = fd
                for i in range(len(fd_value[m]) - 1, -1, -1):
                    if fd_value[m][i] != -1:
                        C.append(fd_value[m][i])
                        fd_value[0].pop(i)
                    if len(fd_value[0]) <= constraints[0]:
                        C = C + [item for item in (fd_value[0] + fd_value[1])]
                        while len(C) < constraints[0] + constraints[1]:
                            for save_center_item in save_center[1]:
                                if save_center_item not in C:
                                    # if i >= constraints[1] - len(C_count[1]):
                                    #     break
                                    C.append(save_center_item)
                                    break
                                    # i += 1
                        right = fd
                        break
            else:
                left = fd
    print(len(C))
    return C

#######################################################################################################################
# constructure the independent set for k groups_ fixed the real value needed5423
def ICS(Gamma_init_X, X, classTable, constraints, m, para, R_dict, left, right, j_bound, metric='euclidean'):
    # each points
    Gamma_x = X  # currently point
    j = j_bound + 1

    right = max(right, np.linalg.norm(np.array(Gamma_x) - np.array(Gamma_init_X[classTable])))

    while left < (para - 1) * (para ** j) <= right:
        R = round((para - 1) * (para ** j), 2)
        if R not in R_dict:
            R_dict[R] = [[] for _ in range(m + 2)]
        Gamma_para = R_dict[R]
        if len(Gamma_para[classTable]) == 0:
            Gamma_para[classTable].append(Gamma_init_X[classTable])
            # first_elements = [Gamma_para[1 - classTable]]

        first_elements = Gamma_para[1 - classTable]

        if min_metric([Gamma_x], Gamma_para[classTable], metric) > R * 2:
            # print(first_elements)
            if classTable == 0:
                Gamma_para[classTable].append(Gamma_x)
            elif min_metric([Gamma_x], first_elements, metric) > R * 3:
                    Gamma_para[classTable].append(Gamma_x)
        if classTable == 1:
            len_gen = len(first_elements)
            if len(Gamma_para[m]) < len_gen:
                Gamma_para[m] = [-1] * len_gen
                Gamma_para[m+1] = [-1] * len_gen
            for empty_ele in range(len_gen):
                if Gamma_para[m][empty_ele] == -1 and min_metric([Gamma_x], [Gamma_para[1 - classTable][empty_ele]],
                                                                 metric) <= R:
                    Gamma_para[m][empty_ele] = Gamma_x
                    Gamma_para[m + 1][empty_ele] = classTable
        if len(Gamma_para[0]) > constraints[0] + constraints[1] or len(Gamma_para[1]) > constraints[1]:
            left = R
            j_bound = j
        j += 1
    return left, right, j_bound


