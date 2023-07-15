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
    return [s, s, w, s, w, w, s, w]


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
    # Initialize the starting state, fringe, and visited nodes.
    current_state = problem.getStartState()
    fringe = util.Stack()
    fringe.push([[current_state]])
    visited_nodes = []

    # Search for the goal state using depth-first search.
    while not fringe.isEmpty():
        current_list = fringe.pop()
        current_state = current_list[-1]

        if current_state[0] not in visited_nodes:
            visited_nodes.append(current_state[0])
            if problem.isGoalState(current_state[0]):
                return [action for _, action, _ in current_list[1:]]

            for next_state, action, cost in problem.getSuccessors(current_state[0]):
                if next_state not in visited_nodes:
                    next_list = current_list + [(next_state, action, cost)]
                    fringe.push(next_list)

    return []  # Return an empty list if the goal state cannot be reached.


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    current_state = problem.getStartState()
    fringe = util.Queue()
    fringe.push([(current_state, None, 0)])
    visited_nodes = []

    while not fringe.isEmpty():
        current_list = fringe.pop()
        current_state = current_list[-1]
        if current_state[0] not in visited_nodes:
            visited_nodes.append(current_state[0])

            if problem.isGoalState(current_state[0]):
                return [action for (_, action, _) in current_list[1:]]

            current_successors = problem.getSuccessors(current_state[0])
            unvisited_nodes = [successor for successor in current_successors
                               if successor[0] not in visited_nodes]

            for neighbour_state in unvisited_nodes:
                tl = current_list + [neighbour_state]
                fringe.push(tl)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    current_state = problem.getStartState()
    fringe = util.PriorityQueue()
    fringe.push([[current_state, None, 0]], 0)
    visited_nodes = []

    while not fringe.isEmpty():
        current_list = fringe.pop()
        current_state = current_list[len(current_list) - 1]

        if current_state[0] not in visited_nodes:
            visited_nodes.append(current_state[0])

            if problem.isGoalState(current_state[0]):
                ans = []
                for i in range(len(current_list) - 1):
                    ts = current_list[i + 1]
                    ans.append(ts[1])
                return ans

            current_successor = problem.getSuccessors(current_state[0])
            unvisited_nodes = []
            for i in range(len(current_successor)):
                if current_successor[i][0] not in visited_nodes:
                    unvisited_nodes.append(current_successor[i])

            current_uniform_cost = 1
            for i in range(len(current_list)):
                current_uniform_cost += current_list[i][2]

            for neighbour_state in unvisited_nodes:
                tl = [] + current_list
                tl.append(neighbour_state)
                tuc = current_uniform_cost + neighbour_state[2]
                fringe.push(tl, tuc)



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    start_state = problem.getStartState()
    start_node = (start_state, None, 0)
    start_cost = 0
    fringe = util.PriorityQueue()
    fringe.push([start_node], start_cost)
    visited = []

    while not fringe.isEmpty():
        current_path = fringe.pop()
        current_node = current_path[-1]

        if current_node[0] in visited:
            continue

        visited.append(current_node[0])

        if problem.isGoalState(current_node[0]):
            return [action for (_, action, _) in current_path][1:]

        for successor in problem.getSuccessors(current_node[0]):
            if successor[0] not in visited:
                new_path = list(current_path)
                new_path.append(successor)
                new_cost = problem.getCostOfActions([action for (_, action, _) in new_path][1:]) + heuristic(
                    successor[0], problem)
                fringe.push(new_path, new_cost)
    return []


def answer(path):
    ans = []
    for _i in range(len(path) - 1):
        ts = path[_i + 1]
        ans.append(ts[1])
    return ans


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
