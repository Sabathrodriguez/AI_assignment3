# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        # print(scores)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # foodDist = 0;
        # for food in newFood.asList():
        #     dist = util.manhattanDistance(newPos, food)
        #     foodDist += util.manhattanDistance(newPos, food)
        #
        # gdist = 0
        # for g in newGhostStates:
        #     gdist += util.manhattanDistance(newPos, g.getPosition())

        if newPos == currentGameState.getPacmanPosition():
            return -10000000000
        return successorGameState.getScore()
        # return min(foodDist, gdist)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    IS_PACMAN = True

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # gameState.getPacmanPosition()
        def minValue(state, agentIndex, depth):
            agentTotalCount = gameState.getNumAgents()
            legalActions = state.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(state)

            if agentIndex == agentTotalCount-1:
                minimumValue = min(maxValue(state.generateSuccessor(agentIndex, action), agentIndex, depth) for action in legalActions)
            else:
                minimumValue = min(minValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth) for action in legalActions)
            return minimumValue


        def maxValue(state, agentIndex, depth):
            agentIndex = 0
            legalActions = state.getLegalActions(agentIndex)

            if not legalActions or depth == self.depth:
                return self.evaluationFunction(state)

            maximumValue = max(minValue(state.generateSuccessor(agentIndex, action), agentIndex+1, depth+1) for action in legalActions)

            return maximumValue

        actions = gameState.getLegalActions(0)
        allActions = {}
        for action in actions:
            allActions[action] = minValue(gameState.generateSuccessor(0, action), 1, 1)

        return max(allActions, key=allActions.get)
        #return MinimaxAgent.value(self, gameState, True, self.depth)[1]
    #def value(self, state, isMaxAgent, depth):


        # if depth == 0:
        #     return self.evaluationFunction(state)
        #
        # if isMaxAgent:
        #     return self.maxValue(state, depth - 1)
        # else:
        #     return self.minValue(state, depth - 1)


    # def maxValue(self, state, depth):
    #     v = -100000000000
    #
    #     legalActions = state.getLegalActions(self.index)
    #     bestAction = None
    #     # Get max value of each successor
    #     for action in legalActions:
    #         nextActionValue = self.value(state.generateSuccessor(self.index, action), False, depth - 1)
    #         if nextActionValue > v:
    #             v = nextActionValue
    #             bestAction = action
    #     return v, bestAction
    #
    #
    # def minValue(self, state, depth):
    #     v = 100000000000
    #     mini = -10000000000
    #
    #     legalActions = state.getLegalActions(self.index)
    #     # Get min value of each successor
    #     for action in legalActions:
    #         mini = self.value(state.generateSuccessor(self.index, action), True, depth - 1)
    #         bestAction = action
    #         if mini < v:
    #             v = mini
    #
    #     return mini




    # def maxValue(self, state, isMaxAgent):
    #     v = -100000000000
    #     if isMaxAgent:
    #         legalActions = state.getLegalActions(0)
    #     else:
    #         legalActions = state.getLegalActions(1)
    #
    #     # Get max value of each successor
    #     for action in legalActions:
    #         if isMaxAgent:
    #             v = max(v, self.value(state.generateSuccessor(0, action), isMaxAgent))
    #         else:
    #             v = max(v, self.value(state.generateSuccessor(1, action), isMaxAgent))
    #
    #         if isMaxAgent:
    #             isMaxAgent = False
    #         else:
    #             isMaxAgent = True
    #
    #     return v

    # def minValue(self, state, isMaxAgent):
    #     v = 100000000000
    #
    #     if isMaxAgent:
    #         legalActions = state.getLegalActions(0)
    #     else:
    #         legalActions = state.getLegalActions(1)
    #
    #     # Get min value of each successor
    #     for action in legalActions:
    #         if isMaxAgent:
    #             v = min(v, self.value(state.generateSuccessor(0, action), isMaxAgent))
    #         else:
    #             v = min(v, self.value(state.generateSuccessor(1, action), isMaxAgent))
    #
    #         if isMaxAgent:
    #             isMaxAgent = False
    #         else:
    #             isMaxAgent = True
    #     return v



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def minValue(state, agentIndex, depth, alpha, beta):
            minimumValue = float('inf')  # I initialize minimumValue as the pseudocode does.
            agentTotalCount = gameState.getNumAgents()
            legalActions = state.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(state)

            for action in legalActions:  # I take this for-loop out so we can update alpha/beta in the for-loop, i.e., update alpha/beta for each action.
                if agentIndex == agentTotalCount - 1:
                    minimumValue = min(minimumValue,
                                       maxValue(state.generateSuccessor(agentIndex, action), 0, depth + 1, alpha, beta)[
                                           0])  # I update the depth here (depth+1) because the instruction says "A single search py is considered to be one Pacman move and all the ghosts' responses." That is, the max layer and min layer should be considered as the same depth, and the depth should be update when moving from min to max. I return the maxValue() function in this getAction() function.

                else:
                    minimumValue = min(minimumValue,
                                       minValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth,
                                                alpha, beta))
                if minimumValue < alpha:
                    return minimumValue
                beta = min(beta, minimumValue)
            return minimumValue

            # return minimumValue

        def maxValue(state, agentIndex, depth, alpha, beta):
            agentIndex = 0
            best_action = None  # Initialize best_action so we can keep track of Pacman's optimal action as we update the value.
            maximumValue = -float('inf')  # Initialize maximumValue as the pseudocode does.
            legalActions = state.getLegalActions(agentIndex)

            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            for action in legalActions:

                new_value = minValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta)
                if new_value > maximumValue:
                    maximumValue = new_value
                    best_action = action  # best_action is updated here.

                if maximumValue > beta:
                    return maximumValue, best_action

                alpha = max(alpha, maximumValue)
            return maximumValue, best_action

        # actions = gameState.getLegalActions(0)
        # allActions = {}
        # for action in actions:
        #    allActions[action] = minValue(gameState.generateSuccessor(0, action), 1, 1, -float('inf'), float('inf'))

        return maxValue(gameState, 0, 0, -float('inf'), float('inf'))[1]  # I return the

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expValue(state, agentIndex, depth):
            agentTotalCount = gameState.getNumAgents()
            legalActions = state.getLegalActions(agentIndex)
            minimumValue = 0

            if not legalActions:
                return self.evaluationFunction(state)

            if agentIndex == agentTotalCount-1:
                for action in legalActions:
                    tempArr = []
                    for i in range(len(legalActions)):
                        tempArr.append(1 / len(legalActions))

                    p = util.getProbability(action, tempArr, legalActions)
                    minimumValue += p * maxValue(state.generateSuccessor(agentIndex, action), agentIndex, depth)
                # minimumValue = p * (maxValue(state.generateSuccessor(agentIndex, action), agentIndex, depth) for action in legalActions)
            else:
                for action in legalActions:
                    tempArr = []
                    for i in range(len(legalActions)):
                        tempArr.append(1/len(legalActions))

                    p = util.getProbability(action, tempArr, legalActions)
                    minimumValue += p * expValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth)
                # minimumValue = p * (expValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth) for action in legalActions)
            return minimumValue


        def maxValue(state, agentIndex, depth):
            agentIndex = 0
            legalActions = state.getLegalActions(agentIndex)

            if not legalActions or depth == self.depth:
                return self.evaluationFunction(state)

            maximumValue = max(expValue(state.generateSuccessor(agentIndex, action), agentIndex+1, depth+1) for action in legalActions)

            return maximumValue

        actions = gameState.getLegalActions(0)
        allActions = {}
        for action in actions:
            allActions[action] = expValue(gameState.generateSuccessor(0, action), 1, 1)

        return max(allActions, key=allActions.get)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
