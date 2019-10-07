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
from operator import itemgetter
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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
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


        "*** YOUR CODE HERE ***"
        food = newFood.asList()
        closeF = None, 0
        closeG = None, 0
        
        if len(food) != 0:
          closeF = min((foo, manhattanDistance(newPos, foo)) for foo in food) #closeF = (closest food, distance to closest food)
        if len(newGhostStates) != 0:
          closeG = min((gho, manhattanDistance(newPos, gho.getPosition())) for gho in newGhostStates) #closeG = (closest ghost, distance to closest ghost)
        
        eval = successorGameState.getScore() # Tracks if it has eaten food or not
        # # Eating food is great --> Nearest food is main focus
        # Near ghost is bad
        # Near ghost is not bad if ghost is scared
        if closeG[0].scaredTimer == 0 and closeG[1] < 3:
          eval = eval - (closeG[1]*25)  #being close to a ghost is really bad
        eval = eval + 1.0/(closeF[1]+1) #+1 so don't ever get zero division error
        return eval

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
        """
        "*** YOUR CODE HERE ***"
        exploredD = 0 # Instantiate explored depth to 0
        def Minimax(state, exploredD):
          if state.isWin() or state.isLose() or exploredD == self.depth * state.getNumAgents(): # Checks terminal or depth
            return self.evaluationFunction(state) 
          agentIndex = exploredD % state.getNumAgents() #Calculates agent index
          if agentIndex == 0: # If pacman return max
            evaledA = {} # Dictionary of key = eval, val = action
            exploredD = exploredD + 1 # Updates explored depth
            for a in state.getLegalActions(agentIndex): # Iterates through legal actions
              tupe = Minimax(state.generateSuccessor(agentIndex, a),exploredD) # Calculates next value / action pair
              evaledA[tupe] = a # Assigns the action to its value key in dict
            maxK = max(evaledA.keys()) # Calculates max key value
            return maxK, evaledA[maxK] # Returns the max key value and its associated action
          else: # Same thing but takes minimum value
            evaledA = {}
            exploredD = exploredD + 1
            for a in state.getLegalActions(agentIndex):
              tupe = Minimax(state.generateSuccessor(agentIndex, a),exploredD)
              evaledA[tupe] = a
            minK = min(evaledA.keys())
            return minK, evaledA[minK]
        ans = Minimax(gameState, exploredD)
        return ans[1] # Returns the action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        exploredD = 0
        
        def maxValue(state, exploredD, alpha, beta):
          if state.isWin() or state.isLose() or exploredD == self.depth * state.getNumAgents(): # Checks terminal or depth
            return self.evaluationFunction(state) 
          v = float("-inf")
          agentIndex = exploredD % state.getNumAgents()
          evaledA = {} # Dictionary of key = eval, val = action
          exploredD = exploredD + 1 # Updates explored depth
          for a in state.getLegalActions(agentIndex): # Iterates through legal actions
            tupe = minValue(state.generateSuccessor(agentIndex, a),exploredD, alpha, beta) # Calculates next value / action pair
            if type(tupe) is float: # Check type of tupe for return type reasons
              evaledA[tupe] = a
            else:
              evaledA[tupe[0]] = a # Assigns the action to its value key in dict
            v = max(evaledA.keys()) # Calculates max key value
            if v > beta: # Checks for beta and returns (prunes) if out of bounds
              return v, evaledA[v]
            if v > alpha: # Updates alpha if new upperbound
              alpha = v
          return v, evaledA[v] # Returns the max key value and its associated action

        def minValue(state, exploredD, alpha, beta):
          if state.isWin() or state.isLose() or exploredD == self.depth * state.getNumAgents(): # Checks terminal or depth
            return self.evaluationFunction(state)
          v = float("inf")
          agentIndex = exploredD % state.getNumAgents()
          evaledA = {} # Dictionary of key = eval, val = action
          exploredD = exploredD + 1 # Updates explored depth
          for a in state.getLegalActions(agentIndex): # Iterates through legal actions
            tupe = ()
            if (agentIndex + 1) % state.getNumAgents() != 0: # If next agent is ghost take min value again
              tupe = minValue(state.generateSuccessor(agentIndex, a),exploredD, alpha, beta) # Calculates next value / action pair
            else:
              tupe = maxValue(state.generateSuccessor(agentIndex, a),exploredD, alpha, beta)
            if type(tupe) is float:
              evaledA[tupe] = a
            else:
              evaledA[tupe[0]] = a # Assigns the action to its value key in dict
            v = min(evaledA.keys()) # Calculates max key value
            if v < alpha: # Checks for alpha and prunes if out of bounds
              return v, evaledA[v]
            if v < beta: # Updates beta if new lower bound
              beta = v
          return v, evaledA[v] # Returns the max key value and its associated action
        
        ans = maxValue(gameState, exploredD, float("-inf"), float("inf"))
        return ans[1] # Returns the action

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
        exploredD = 0
        def Minimax(state, exploredD):
          if state.isWin() or state.isLose() or exploredD == self.depth * state.getNumAgents(): # Checks terminal or depth
            return self.evaluationFunction(state) 
          agentIndex = exploredD % state.getNumAgents() #Calculates agent index
          if agentIndex == 0: # If pacman return max
            evaledA = {} # Dictionary of key = eval, val = action
            exploredD = exploredD + 1 # Updates explored depth
            for a in state.getLegalActions(agentIndex): # Iterates through legal actions
              tupe = Minimax(state.generateSuccessor(agentIndex, a),exploredD) # Calculates next value / action pair
              evaledA[tupe] = a # Assigns the action to its value key in dict
            maxK = max(evaledA.keys()) # Calculates max key value
            if exploredD == 1: # Only returns the key value pair if its the first call
              return maxK, evaledA[maxK] # Returns the max key value and its associated action
            return maxK
          else: # Same thing but takes minimum value
            evaledA = {}
            exploredD = exploredD + 1
            for a in state.getLegalActions(agentIndex):
              tupe = Minimax(state.generateSuccessor(agentIndex, a),exploredD) 
              evaledA[tupe] = a # No reason to assign to dictionary, a list of evales will do but i'm too lazy to change it
            avgK = float(sum(evaledA.keys()) / len(evaledA.keys())) #Calculates avgK
            return avgK
        ans = Minimax(gameState, exploredD)
        return ans[1] # Returns the action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    food = currentGameState.getFood().asList() #food as a list of food coordinates
    pos = currentGameState.getPacmanPosition() #paman's current position
    closeGhost = 0 #Penalty for threateningly close ghost

    #of the ghosts in the game, they are relevant to evaluation if they are threateningly close and are not scared
    if len(currentGameState.getGhostStates()) != 0: 
      #closest ghost
      closeGhost = min(manhattanDistance(ghost.getPosition(), pos) for ghost in currentGameState.getGhostStates())
    #f(closeGhost != 0):
    #  closeGhost = 20/(closeGhost+1)
    distFood = min(0, (manhattanDistance(foo,pos) for foo in food))
    return currentGameState.getScore() - closeGhost + 100/(len(food)+1) + 70/(distFood+1)

# Abbreviation
better = betterEvaluationFunction

