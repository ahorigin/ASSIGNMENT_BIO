import random

import numpy as np
import pandas as pd
import numpy.random as npr
import util
from matplotlib import pyplot as plt

NUM_OF_GENES = 32
GENE_SIZE = 6
NUM_GENES = NUM_OF_GENES * GENE_SIZE
NUM_OF_INDI = 10
NUM_GENERATION = 1000
GENE_SIZE = 6
M_RATE = 0.1

def tournment_selection(population):

    p1 = population[np.random.randint(0,NUM_OF_INDI)]
    p2 = population[np.random.randint(0,NUM_OF_INDI)]
    while p1 == p2:
        p1 = population[np.random.randint(0, NUM_OF_INDI)]
        p2 = population[np.random.randint(0, NUM_OF_INDI)]
    if p1.fitness > p2.fitness:
        return p1
    else:
        return p2



def sort(pop):
    # bubble sort algorihtm to be used in rank selection
    n = len(pop)
    for i in range(n):
        already_sorted = True
        for j in range(n - i - 1):
            if pop[j].fitness > pop[j + 1].fitness:
                pop[j], pop[j + 1] = pop[j + 1], pop[j]
                already_sorted = False
        if already_sorted:
            break
    return pop

def cross_over(parents, type = None):
    #
    p1 = parents[0]
    p2 = parents[1]
    if type == 1: # single point cross over
        cr_point = random.randint(1, NUM_OF_GENES)
        p1_ = split_into_individual_genes(p1.rule)
        p2_ = split_into_individual_genes(p2.rule)
        ind = Individual("".join(p1_[:cr_point]+p2_[cr_point:]))

    elif type == 2: # double point cross over
        # p1 = population[0]
        # p2 = population[1]
        cr_point_1 = random.randint(1, 32)
        cr_point_2 = random.randint(1, 32)
        while True:
            if cr_point_1 >= cr_point_2:
                cr_point_1 = random.randint(1, 32)
                cr_point_2 = random.randint(1, 32)
            else:
                break

        p1_ = split_into_individual_genes(p1.rule)
        p2_ = split_into_individual_genes(p2.rule)
        p1_1 = p1_[:cr_point_1]
        p1_2 = p1_[cr_point_1:cr_point_2]
        p1_3 = p1_[cr_point_2:]

        p2_1 = p2_[:cr_point_1]
        p2_2 = p2_[cr_point_1:cr_point_2]
        p2_3 = p2_[cr_point_2:]
        # print(p1.rule)
        # print(p2.rule)
        # print("".join(p1_1+p2_2+p1_3))
        # p1__ = []
        # while len(p1__) < len(p1_1):
        #     for i in
        p1__ = []
        p3__  = []

        for k in  p1_:
            if k not in p2_2:
                if len(p1__) != len(p1_1):
                    p1__.append(k)
        for k in  p1_:
            if k not in p2_2:
                if len(p3__) != len(p2_3):
                    p3__.append(k)
        ind = Individual("".join(p1_1 + p2_2 + p1_3))
        # ind = Individual("".join(p1__+p2_2+p3__))



    return ind

class Individual():
    rule = None
    fitness = 0

    def __init__(self, rule):
        # print(rule)
        self.rule = "".join(str(item) for item in rule)
        self.fitness = 0

    def reset(self):
        self.fitness = 0
        # print(chr_R, chr_C)


def split_into_individual_genes(rule):
    # function to split the individual into individual genes
    return [rule[i:i + GENE_SIZE] for i in range(0, len(rule), GENE_SIZE)]

def get_best_worst(population,worst=None):
    # function that find best and worst individual on the population based on the fitness score
    if worst:
        # print([a.fitness for a in population])
        fitness = 100
        for i in population:
            if i.fitness < fitness:
                fitness = i.fitness
        return fitness
    fitness = 0
    for i in population:
        if i.fitness > fitness:
            fitness = i.fitness
    return fitness



def calfitness(train, ind):
    # function to calculate the fitness score of the individual
    fitness = 0
    for k in list(set(split_into_individual_genes(ind.rule))):
        for j in train:
            # print(k, "==", j)
            if k == j:
                fitness += 1
                break
    return (fitness / 32)*100

def get_avg_fitness(pop):
    # function to get the avg fitness of the population
    total = 0
    for i in pop:
        total =+ i.fitness
    return total/NUM_OF_INDI





def select(population):
    # selection based on roulette wheel
    max = sum([i.fitness for i in population])
    selection_probs = [(c.fitness / max) for c in population]
    p1 = population[npr.choice(len(population), p=selection_probs)]
    # pop = sort(population)
    # n = len(pop)
    # rank_sum = n * (n + 1) / 2
    # probs = []
    # for rank, ind_fitness in enumerate(sort(pop),1):
    #     # print(rank, ind_fitness.fitness, float(rank) / rank_sum)
    #     probs.append(float(rank) / rank_sum)
    # # print(probs)
    # p1 = pop[npr.choice(len(pop), p=probs)]

    print("ss",p1.fitness)
    return p1



def mutation(offs):

    s = split_into_individual_genes(offs.rule)
    rd = random.randint(0,191)
    s = list(offs.rule)
    # s = s[rd]

    if s[rd] == "0":
        s[rd] = "1"
    else:
        s[rd] = "0"

    offs_ = Individual("".join(s))
    offs_.fitness = calfitness(train, offs_)
    return offs_

# this section of code is used for testing purpose of inital populations performance
pop = []
for i in range(5):
    ind = []
    for j in range(25):
        gene = list(np.random.randint(2, size=6))
        if gene not in ind:
            ind.append("".join(["".join(str(k)) for k in gene]))
    pop.append(Individual("".join(ind)))
population = pop

df = pd.read_csv("data1 .txt").iloc[:, 0].str.split(" ", expand=True)
df.columns = ["input1", "output"]
df = df.input1 + df.output

train, test = util.split_into_train_test(df, 0.6)

population = [Individual(np.random.randint(2, size=NUM_GENES)) for i in range(NUM_OF_INDI)]
for k in population:
    k.fitness = calfitness(train,k)
    print(k.fitness)
worst = []
best = []
avg_fitness = []
gen = []
# random.seed(0)
for l  in range(NUM_GENERATION):
    print(" ", l)
    print(sum([i.fitness for i in population])/NUM_OF_INDI)
    avg_fitness.append(sum([i.fitness for i in population])/NUM_OF_INDI)
    best.append(get_best_worst(population))
    worst.append(get_best_worst(population,worst=True))

    gen.append(l)
    print([i.fitness for i in population])
    pop = []

    mt = random.uniform(0, 1)

    while len(pop) < NUM_OF_INDI:

        # p1 = select(population)
        # p2 = select(population)
        p1 = tournment_selection(population)
        p2 = tournment_selection(population)

        # print("Selected:",p1.fitness, p2.fitness)
        p1_genes = split_into_individual_genes(p1.rule)
        p2_genes = split_into_individual_genes(p2.rule)

        # for p, j in enumerate(p1_genes):
        # cross_point = random.randint(0,191)
        # offs1 = p1.rule[:cross_point] + p2.rule[cross_point:]
        # offs2 = p1.rule[cross_point:] + p2.rule[:cross_point]
        #
        # o1 = Individual(offs1)
        # o2 = Individual(offs2)

        # o1.fitness = calfitness(train,o1)
        # o2.fitness = calfitness(train,o2)
        #
        # print("c",o1.fitness, o2.fitness)
        #
        # if o1.fitness > o2.fitness:
        #     to_add = o1
        # else:
        #     to_add = o2
        # print(p1.rule)
        # print(p2.rule)

        ind = cross_over([p1,p2],1)
        ind.fitness = calfitness(train,ind)
        # # else:
        # pop.append(p1)
        # pop.append(p2)
        if mt > M_RATE:
            pop.append(mutation(ind))
        else:
            pop.append(ind)

    population = pop
    pop = []

de =  pd.DataFrame({"Num of Generations":gen,"Avg":avg_fitness,"best":best,"worst":worst})
de.plot(title= f"Data1.txt\n ( mutation rate {M_RATE} , Train data size: {train.shape[0]},\n Number of generation :{NUM_GENERATION} ",y=["Avg","worst","best"],x="Num of Generations",kind="line")
plt.show()


# this part of code was used to test the final solution with test set.
checked = []
for i in population:
    s = split_into_individual_genes(i.rule)
    for b in s:
        if b in test.to_list() and b not in checked:
            checked.append(b)




