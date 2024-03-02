import robby
#import numpy as np
from utils import *
import random
import matplotlib.pyplot as plt
POSSIBLE_ACTIONS = ["MoveNorth", "MoveSouth", "MoveEast", "MoveWest", "StayPut", "PickUpCan", "MoveRandom"]
rw = robby.World(10, 10)
rw.graphicsOff()


def sortByFitness(genomes):
    tuples = [(fitness(g), g) for g in genomes]
    tuples.sort()
    sortedFitnessValues = [f for (f, g) in tuples]
    sortedGenomes = [g for (f, g) in tuples]
    return sortedGenomes, sortedFitnessValues


def randomGenome(length):
    """
    :param length:
    :return: string, random integers between 0 and 6 inclusive
    """

    """Your Code Here"""
    # raiseNotDefined()
    genome = ""
    for n in range(length):
        genome += str(random.randint(0,6))
    return genome


def makePopulation(size, length):
    """
    :param size - of population:
    :param length - of genome
    :return: list of length size containing genomes of length length
    """


    """Your Code Here"""
    # raiseNotDefined()
    popList = []
    for n in range(size):
        popList.append(randomGenome(length))
    return popList

def fitness(genome, steps=200, init=0.50):
    """

    :param genome: to test
    :param steps: number of steps in the cleaning session
    :param init: amount of cans
    :return:
    """
    if type(genome) is not str or len(genome) != 243:
        raise Exception("strategy is not a string of length 243")
    for char in genome:
        if char not in "0123456":
            raise Exception("strategy contains a bad character: '%s'" % char)
    if type(steps) is not int or steps < 1:
        raise Exception("steps must be an integer > 0")
    if type(init) is str:
        # init is a config file
        rw.load(init)
    elif type(init) in [int, float] and 0 <= init <= 1:
        # init is a can density
        rw.goto(0, 0)
        rw.distributeCans(init)
    else:
        raise Exception("invalid initial configuration")

    # raiseNotDefined()

    # 25 sessions?
    fitness = 0
    for x in range(25):
        rw.goto(0, 0)
        rw.distributeCans(init)
        for i in range(steps):
            # if genome[i] == "1":
            #     fitness += 1
            #  rw.getCurrentPosistion()
            # rw.
            fitness += rw.performAction(POSSIBLE_ACTIONS[int(genome[rw.getPerceptCode()])])
    return (fitness/25)

def evaluateFitness(population):
    """
    :param population:
    :return: a pair of values: the average fitness of the population as a whole and the fitness of the best individual
    in the population.
    """
    # raiseNotDefined()
    bestFitness = -9999
    avgFitness = 0
    
    for i in range(len(population)):
        newFitness = fitness(population[i])
        if newFitness > bestFitness:
            bestFitness = newFitness
        avgFitness += newFitness
    return (avgFitness/len(population), bestFitness)


def crossover(genome1, genome2):
    """
    :param genome1:
    :param genome2:
    :return: two new genomes produced by crossing over the given genomes at a random crossover point.
    """
    # raiseNotDefined()
    crossPt = random.randint(1,len(genome1)-1)
    cross1start = genome1[:crossPt]
    cross2start = genome2[:crossPt]

    cross1end = genome1[crossPt:]
    cross2end = genome2[crossPt:]

    child1 = cross1start + cross2end
    child2 = cross2start + cross1end
    return (child1, child2)


def mutate(genome, mutationRate):
    """
    :param genome:
    :param mutationRate:
    :return: a new mutated version of the given genome.
    """
    # raiseNotDefined()
    for i in range(len(genome)):
        chance = random.random()
        if chance <= mutationRate:
            newGene = random.randint(0,6)
            while newGene == genome[i]:
                newGene = random.randint(0,6)
            genome = genome[:i] + str(newGene) + genome[i+1:]
    return genome

def selectPair(population):
    """

    :param population:
    :return: two genomes from the given population using fitness-proportionate selection.
    This function should use RankSelection,
    """
    # raiseNotDefined()

    #Need to use RankSelection

    weights = []
    dic = {}
    for n in range(len(population)):
        # weights.append(fitness(n))
        # dic[n] = fitness(n)
        weights.append(n)

    # print(dic)
    # rankedFitness = sorted(dic.items(), key=lambda x:x[1])
    # print(rankedFitness)
    # (population, fit) = sortByFitness(population)

    choice1 = weightedChoice(population,weights)
    # i = population.index(choice1)
    # print(choice1)
    # print(i)
    # newPop = population[:i] + population[i+1:]
    # newWeights = weights[:i] + weights[i+1:]
    # for p in population:
    #     if p is not choice1:
    #         newPop.append(p)
    #         weights.append(fitness(p))
    # print(newPop)
    # choice2 = weightedChoice(newPop,newWeights)
    choice2 = weightedChoice(population,weights)
    return (choice1, choice2)

xPts = []
yPts = []
xList = []
yList = []
bestStrat = ""
absBestFit = -9999
def runGA(populationSize, crossoverRate, mutationRate, logFile=""):
    """

    :param populationSize: :param crossoverRate: :param mutationRate: :param logFile: :return: xt file in which to
    store the data generated by the GA, for plotting purposes. When the GA terminates, this function should return
    the generation at which the string of all ones was found.is the main GA program, which takes the population size,
    crossover rate (pc), and mutation rate (pm) as parameters. The optional logFile parameter is a string specifying
    the name of a te
    """
    # raiseNotDefined()
    if logFile != "":
            # print("File output: "+logFile)
            file = open(logFile,'w')
    print("Population size: "+ str(populationSize))
    genomeLength = 243
    print("Genome length: " + str(genomeLength))
    population = makePopulation(populationSize,genomeLength)
    # global bestFile
    global bestStrat
    global absBestFit
    global xPts
    global yPts
    global xList
    global yList
    overallBestFit = -9999
    overallBestGen = 0
    overallAvgFit = -9999
    overallBestStrat = ""

    for i in range(300):
        (population, fit) = sortByFitness(population)
        (avgFit, bestFit) = evaluateFitness(population)
        print("Generation {}: \taverage fitness {}, \tbest fitness {}".format(i,round(avgFit,4),bestFit))
        xPts.append(i)
        yPts.append(bestFit)
        if bestFit >= overallBestFit:
            # if avgFit > overallAvgFit:
            overallBestFit = bestFit
            overallAvgFit = avgFit
            overallBestGen = i
            overallBestStrat = population[len(population)-1]
            # xList.append(xPts)
            # yList.append(yPts)
            # xPts = []
            # yPts = []
        if bestFit >= absBestFit:
            bestFile = open("bestStrategy.txt", 'w')
            absBestFit = bestFit
            bestFile.write("{}\n".format(population[len(population)-1]))
            # bestStrat = population[len(population)-1]
        # if bestFit == genomeLength:
        #     if logFile != "":
        #         file.write("{} {} {}\n".format(i,avgFit,bestFit))
        #     break
        # crossList = []
        childrenList = []
        # for p in range(0,len(population),2):

        #     (par1, par2) = selectPair(population)
        #     crossList.append(par1)
        #     crossList.append(par2)

        for p in range(0,len(population),2):
            cross = random.random()
            (par1, par2) = selectPair(population)
            if cross <= crossoverRate:
                # print("Picking parents")
                
                # print("p: "+str(par1))
                # print("Crossover")
                (child1, child2) = crossover(par1, par2)
                # print("p: "+str(child1))
                child1 = mutate(child1, mutationRate)
                child2 = mutate(child2, mutationRate)
                childrenList.append(child1)
                childrenList.append(child2)
            else:
                # (par1, par2) = selectPair(crossList)
                child1 = mutate(par1, mutationRate)
                child2 = mutate(par2, mutationRate)
                childrenList.append(child1)
                childrenList.append(child2)
        # print(len(childrenList))
        population = childrenList
        # print(len(population))
        if logFile != "" and (i+1) % 10 == 0:
            file.write("{}\t{}\t{}\t{}\n".format(overallBestGen, round(overallAvgFit,4), overallBestFit, overallBestStrat))
            # print("Write: {} {} {} {}\n".format(overallBestGen, overallAvgFit, overallBestFit, overallBestStrat))
            overallBestFit = -9999
            overallBestGen = 0
            overallAvgFit = -9999
            overallBestStrat = ""
    xList.append(xPts)
    yList.append(yPts)
    xPts = []
    yPts = []

def test_FitnessFunction():
    f = fitness(rw.strategyM)
    print("Fitness for StrategyM : {0}".format(f))

test_FitnessFunction()

# population = makePopulation(100,243)
# print(selectPair(population))
# print(sortByFitness(population))
# runGA(100, 1.0, 0.05)
# for i in range(5):
runGA(100, .75, 0.005,"run2.txt")

# rList = []
# for i in range(5):
#     r = random.randint(0,len(xList))
#     while r in rList:
#         r = random.randint(0,len(xList))
#     rList.append(r)
for i in range(len(xList)):
    plt.plot(xList[i],yList[i])
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.show()
