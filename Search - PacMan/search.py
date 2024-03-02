# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # Save start state
    start = problem.getStartState()
    print("Start:", start)
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # If start is goal then end
    if problem.isGoalState(start):
        return []
    # Nodes to expand from start
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    # Directions to goal list
    solutionList = []

    # DFS stack of nodes to search and expand
    DFSstack = util.Stack()
    # Add start node to stack so its not empty
    DFSstack.push((start, 'Undefined', 0))

    # List of nodes where DFS has already gone and not expand again
    visited = {}
    # visited[start] = 'Undefined'
    parents = {} # Dictionary of solutions parent nodes when building path
    goalReached = False
    while DFSstack.isEmpty() != True and goalReached != True:
        currentNode = DFSstack.pop()
        visited[currentNode[0]] = currentNode[1] # Save direction of current node from last

        if problem.isGoalState(currentNode[0]):
            goalReached = True
            nodeSolution = currentNode[0]
            break
        for n in problem.getSuccessors(currentNode[0]):
            if n[0] not in visited.keys():
                parents[n[0]] = currentNode[0]
                DFSstack.push(n)

    while nodeSolution in parents.keys():
        solParent = parents[nodeSolution]
        solutionList.insert(0, visited[nodeSolution])
        nodeSolution = solParent
    return solutionList

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"   
    # util.raiseNotDefined()
    # Save start state
    start = problem.getStartState()
    print("Start:", start)
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # If start is goal then end
    if problem.isGoalState(start):
        return []
    # Nodes to expand from start
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    # Directions to goal list
    solutionList = []

    # DFS stack of nodes to search and expand
    BFSQueue = util.Queue()
    # Add start node to stack so its not empty
    BFSQueue.push((start, 'Undefined', 0))

    # List of nodes where DFS has already gone and not expand again
    visited = {}
    # visited[start] = 'Undefined'
    parents = {} # Dictionary of solutions parent nodes when building path
    goalReached = False

    while BFSQueue.isEmpty() != True and goalReached != True:
        currentNode = BFSQueue.pop()
        visited[currentNode[0]] = currentNode[1] # Save direction of current node from last
        if problem.isGoalState(currentNode[0]):
            goalReached = True
            nodeSolution = currentNode[0]
            break
        for n in problem.getSuccessors(currentNode[0]):
            if n[0] not in visited.keys() and n[0] not in parents.keys():
                parents[n[0]] = currentNode[0]
                BFSQueue.push(n)

    while nodeSolution in parents.keys():
        solParent = parents[nodeSolution]
        solutionList.insert(0, visited[nodeSolution])
        nodeSolution = solParent
    return solutionList

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    start = problem.getStartState()
    print("Start:", start)
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # If start is goal then end
    if problem.isGoalState(start):
        return []
    # Directions to goal list
    solutionList = []

    # DFS stack of nodes to search and expand
    UCSPriQueue = util.PriorityQueue()
    # Add start node to stack so its not empty
    UCSPriQueue.push((start, 'Undefined', 0),0)

    # List of nodes where DFS has already gone and not expand again
    visited = {}
    visited[start] = 'Undefined'
    parents = {} # Dictionary of solutions parent nodes when building path
    goalReached = False
    costs = {}
    costs[start] = 0

    while UCSPriQueue.isEmpty() != True and goalReached != True:
        currentNode = UCSPriQueue.pop()
        visited[currentNode[0]] = currentNode[1] # Save direction of current node from last
        if problem.isGoalState(currentNode[0]):
            goalReached = True
            nodeSolution = currentNode[0]
            break
        for n in problem.getSuccessors(currentNode[0]):
            if n[0] not in visited.keys():
                costTotal = currentNode[2] + n[2]
                if n[0] in costs.keys():
                    # print(costs[n[0]])
                    # print(costTotal)
                    if costs[n[0]] <= costTotal:
                        print("Doing Nothing")
                    else:
                        parents[n[0]] = currentNode[0]
                        UCSPriQueue.push((n[0], n[1], costTotal),costTotal)
                        costs[n[0]] = costTotal
                else:
                    parents[n[0]] = currentNode[0]
                    UCSPriQueue.push((n[0], n[1], costTotal),costTotal)
                    costs[n[0]] = costTotal
    # print(parents)
    while nodeSolution in parents.keys():
        solParent = parents[nodeSolution]
        solutionList.insert(0, visited[nodeSolution])
        nodeSolution = solParent
    return solutionList

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    start = problem.getStartState()
    print("Start:", start)
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # If start is goal then end
    if problem.isGoalState(start):
        return []
    # Nodes to expand from start
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    # Directions to goal list
    solutionList = []

    # DFS stack of nodes to search and expand
    UCSPriQueue = util.PriorityQueue()
    # Add start node to stack so its not empty
    UCSPriQueue.push((start, 'Undefined', 0),0)

    # List of nodes where DFS has already gone and not expand again
    visited = {}
    visited[start] = 'Undefined'
    parents = {} # Dictionary of solutions parent nodes when building path
    goalReached = False
    costs = {}
    costs[start] = 0

    while UCSPriQueue.isEmpty() != True and goalReached != True:
        currentNode = UCSPriQueue.pop()
        visited[currentNode[0]] = currentNode[1] # Save direction of current node from last
        if problem.isGoalState(currentNode[0]):
            goalReached = True
            nodeSolution = currentNode[0]
            break
        for n in problem.getSuccessors(currentNode[0]):
            if n[0] not in visited.keys():
                costTotal = currentNode[2] + n[2] + heuristic(n[0], problem)
                if n[0] in costs.keys():
                    # print(costs[n[0]])
                    # print(costTotal)
                    if costs[n[0]] <= costTotal:
                        print("Doing Nothing")
                    else:
                        parents[n[0]] = currentNode[0]
                        UCSPriQueue.push((n[0], n[1], currentNode[2] + n[2]),costTotal)
                        costs[n[0]] = costTotal
                else:
                    parents[n[0]] = currentNode[0]
                    UCSPriQueue.push((n[0], n[1], currentNode[2] + n[2]),costTotal)
                    costs[n[0]] = costTotal
    # print("TEST")
    # print(parents)
    while nodeSolution in parents.keys():
        solParent = parents[nodeSolution]
        solutionList.insert(0, visited[nodeSolution])
        nodeSolution = solParent
    return solutionList

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
