import os
from time import time as time
import numpy as np
import pandas as pd
import csv
import math
from scipy.spatial import distance
from fiveApprox import fiveApprox
from fourApprox import fourApprox
from chip import chip
from collections import Counter

from copy import deepcopy

numIter = 2  # Number of iterations
percentEachGroup = 0.0001   # Percentage of member for each group
dataset = 'athlete'


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def norm():
    df = pd.read_csv(PATH_o,header = None)
    features = df.iloc[:, :-1] 
    labels = df.iloc[:, -1] 
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    normalized_df = pd.DataFrame(normalized_features, columns=features.columns)
    normalized_df[''] = labels 
    print(normalized_df.head()) 
    normalized_df.to_csv(PATH, index=False, header = False) 

# four Approx
def evaluate(X, C):
    min_distances = np.min(distance.cdist(X, np.array(C)), axis=1)
    return max(min_distances)


import re


def extract_constraints(file_path, target_filename):
    constraints_list = []

    pattern = re.compile(
        r'path:\s*(?P<path>[^;]+);\s*constraints:\s*\[(?P<constraints>[^\]]+)\]'
    )

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                path = match.group('path').strip()
                constraints_str = match.group('constraints').strip()

                if path == target_filename:
                    constraints = [int(num) for num in constraints_str.split()]
                    break

    return constraints

def extract_constraints_real(file_path):

    df = pd.read_csv(file_path) 
    last_column = df.iloc[:, -1]
    counts = last_column.value_counts()
    return counts


############## real world dataset: approximation ratio #####################
PATH_o = "YoutFilePath"
PATH = "OutputYoutFilePath"
def Simu_approx(iterN):
    for r in range(1):
        counts_m = extract_constraints_real(PATH)
        data_lines = np.sum(counts_m)
        for rr in range(1,11):
            for rounds in range(iterN):
                totalTime = []
                totalLoss = []
                constraints = [math.ceil(x * percentEachGroup * rr) for x in counts_m]
                print(constraints)
                # 4-Approx
                start = time()
                C = fourApprox(constraints, 0.1, PATH)
                totalTime.append(time() - start)
                
                max_cost = 0
                with open(PATH, 'r') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    next(csv_reader)
                    for row in csv_reader:
                        X = [float(item) for item in row[:-1]]
                        cost = evaluate([X], C)
                        max_cost = max(max_cost, cost)
                
                totalLoss.append(max_cost)
               
                # # 5-Approx
                start = time()
                C = fiveApprox(constraints, 0.1, PATH)
                totalTime.append(time() - start)
                
                max_cost = 0
                with open(PATH, 'r') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    next(csv_reader)
                    for row in csv_reader:
                        X = [float(item) for item in row[:-1]]
                        cost = evaluate([X], C)
                        max_cost = max(max_cost, cost)
                totalLoss.append(max_cost)
               
                n_star = [percentEachGroup * rr] * len(totalLoss)
                data_star = [dataset] * len(totalLoss)
                data = {
                    'dataset': data_star,
                    'alg': ['5-Approx'],
                    'approx': totalLoss,
                    'runtime': totalTime,
                    'percent': n_star
                }
                print(data)

         
                df = pd.DataFrame(data)

                file_exists = os.path.isfile('output/real_data_exp.csv')
                df.to_csv('output/real_data_exp.csv', mode='a', index=False, header=not file_exists)

norm()

Simu_approx(numIter)

def wrapperFour(iterN):
    for r in range(1900, 9900, 2000):
        PATH = "Your_Path"
        for rr in range(10):
            PATH1 = 'Your_Path_1.txt'
            for rounds in range(iterN):

                totalTime = []
                totalLoss = []
                constraints = [2,2]
                print(constraints)
                for i in range(iterN):
                    start = time()
                    C = fourApprox(constraints, 0.1, PATH,PATH1)
                    elapse = time() - start
                    max_cost = 0

                    with open(PATH, 'r') as csvfile:
                        csv_reader = csv.reader(csvfile)
                        next(csv_reader)
                        for row in csv_reader:
                            X = [float(item) for item in row]
                            cost = evaluate([X], C)
                            max_cost = max(max_cost, cost)

                    totalLoss.append(max_cost/0.5)
                    totalTime.append(elapse)

                data = {
                    'alg': ['Algorithm1', 'Algorithm2', 'Algorithm3'],
                    'approx': [1.5, 2.0, 1.8],
                    'runtime': [100, 200, 150],
                    '#n': [1000, 2000, 1500]
                }

                df = pd.DataFrame(data)
                df.to_csv('output.csv', index=False)

                with open('output_athlete.txt', 'a') as file:
                    for i in range(0, 1):
                        mean_loss = np.mean(totalLoss)
                        print(f"Mean loss: {mean_loss}")
                        file.write(f'cost: {totalLoss}\n')
                        file.write(f'mean loss: {mean_loss},\n')

                        mean_time = np.mean(totalTime)
                        print(f"Mean runtime: {mean_time}")
                        file.write(f'runtime: {totalTime}\n')
                        file.write(f'mean runtime: {mean_time},\n')


