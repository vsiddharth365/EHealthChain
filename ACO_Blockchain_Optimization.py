import time
import bisect
from variableInitialization import *

solution = [0]  # solution that the algorithm will find
startTime, endTime = 0, 0
pheromone = []  # list to pheromones released by all ants
probability = []  # probability that an ant is considered as a solution
taboo = []  # taboo table to keep track of evaluated solutions
cumulativeProbability = []  # cumulative probability for roulette wheel selection
alpha = 0.2  # heuristic parameter that controls the effect of the pheromone quantity
beta = 2  # heuristic parameter that determines the importance of the solution's quality
rho = 0.1  # constant that represents the rate of pheromone evaporation, which simulates the effect of the evaporation of the pheromones at each step.
rhog = 0.2  # global evaporation rate
utilityOnLowestNV = []  # blockchain utility on lowest values of N and V
utilityOnHighestNV = []  # blockchain utility on highest values of N and V
utilityOnSolution = []  # blockchain utility on solution


# function to find the minimum non-negative integer not present in a list of non-negative integers
def findMex(lis):
    n = len(lis)  # get the length of list
    s = set(range(n + 2))  # create a set of integers from 0 to n+1 inclusive
    for i in range(n):
        if 0 <= lis[i] <= n + 1:  # remove each element of the list from the set
            s.discard(lis[i])
    return min(s)  # minimum integer remaining in set is the answer


# function to execute ACO algorithm on a given ant population and their fitness values
def aco(ant, fitness):
    iteration = 0  # set the iteration to 0
    maxIteration = 100  # set the maximum iterations to 100
    if len(fitness) > 1:
        bestAnt = np.random.randint(0, len(fitness) - 1)  # assume one of the ants to be having the best fitness in the population
    else:
        bestAnt = 0
    taboo.append(bestAnt)  # append the index of selected best ant to the taboo table
    for a in range(len(fitness)):  # for each ant int population
        pheromone.append(fitness[a] / (sum(fitness) / len(fitness)))  # initialize its pheromone value as the ratio of its fitness to average fitness
        probability.append(0)  # intialize the probability with 0
        cumulativeProbability.append(0)  # cumulative probability remains 0
    for iteration in range(maxIteration):  # run the iterations
        for a in range(len(ant)):  # for each ant in population
            pr = ((pheromone[a] ** alpha) * ((1 / fitness[a]) ** beta)) / (
                sum([pheromone[x] ** alpha * (1 / fitness[x]) ** beta for x in range(len(fitness))]))  # calculate the probaility of ant 'a' being selected as the solution
            probability[a] = pr
        prob = 0  # set this to 0 to find the cumulative probability
        for i in range(len(probability)):
            prob += probability[i]
            cumulativeProbability[i] = prob  # get the cumulative probability
        r = np.random.uniform(0, 1)  # random float between 0 and 1 inclusive
        currentAnt = bisect.bisect_left(cumulativeProbability, r, 0, len(cumulativeProbability))  # apply roulette wheel selection to choose an ant
        oldBestFitness = fitness[bestAnt]  # get the best fitness of the previous iteration
        if currentAnt == len(cumulativeProbability):  # rectify the index of the chosen ant if it greater than the largest possible index
            currentAnt = len(cumulativeProbability) - 1
        if fitness[currentAnt] < fitness[bestAnt]:  # if the fitness of currently chosen ant is lesser than the old best fitness
            if currentAnt in taboo:  # if the current ant has already been chosen, known by its presence in taboo table
                if len(taboo) < len(ant):  # if all ants have not yet been chosen
                    currentAnt = findMex(taboo)  # select the ant whose index is the minimum non-negative integer not present in taboo table
            else:
                bisect.insort(taboo, currentAnt)  # add the current ant's index to the taboo table, keeping it sorted
            bestAnt = currentAnt  # update the index of best ant
            if oldBestFitness != 0 and abs(fitness[bestAnt] - oldBestFitness) / abs(oldBestFitness) <= 1 / 1e30:  # condition to break the loop
                break
            for a in range(len(ant)):  # for each ant in population
                pheromone[a] = (1 - rhog) * pheromone[a] + pheromone[a] * (1 / fitness[bestAnt])  # update the pheromone using global rate of evaporation 'rhog'
        elif fitness[currentAnt] == fitness[bestAnt]:  # if fitness of current ant is equal to previous best fitness, break the loop
            break
        else:  # if the fitness of current ant is not better than that of the best ant
            for a in range(len(ant)):  # for each ant in population
                pheromone[a] = (1 - rho) * pheromone[a] + rho * (1 / fitness[a])  # update its pheromone using rate of evaporation 'rho'
    taboo.clear()  # clear the lists to free up auxiliary space
    pheromone.clear()
    probability.clear()
    cumulativeProbability.clear()
    print(f"Total number of iterations to obtain the solution by ACO is {iteration + 1}")
    return ant[bestAnt]  # return the best ant


# function to export the solution obtained, start and end time taken by ACO algorithm to solve blockchain utility objective
def export_ACO_Blockchain_Metrics():
    return [solution, startTime, endTime]


# helper function to run ACO algorithm for blockchain optimization
def ACO_BlockchainOptimization():
    global startTime, endTime, solution
    startTime = time.time()  # get the start time of the algorithm
    ant = initializeBlockchainWhalePopulation()  # get the ant population, which is same as the whale population in NGS-WOA algorithm for blockchain optimization
    solution = aco(ant, calculateBlockchainWhaleFitness(ant))  # get the solution of ACO algorithm
    endTime = time.time()   # get the end time of the algorithm
    print("\nTime taken to obtain blockchain optimization solution in ACO: ", endTime - startTime)
    print(f"The optimal number of transactions per block and validators are {solution[0]} and {solution[1]} respectively")
