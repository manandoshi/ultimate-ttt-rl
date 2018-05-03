from ultimateboard import UTTTBoardDecision, UTTTBoard, stateToNP
from learning import TableLearning
import random
import numpy as np


class UTTTPlayer(object):
    def __init__(self):
        self.board = None
        self.player = None

    def setBoard(self, board, player):
        self.board = board
        self.player = player   # X or O
        self.playernum = 1 if player == 'X' else -1

    def isBoardActive(self):
        return (self.board is not None and self.board.getBoardDecision() == UTTTBoardDecision.ACTIVE)

    def makeNextMove(self, learn=True):
        raise NotImplementedError

    def learnFromMove(self, prevBoardState):
        raise NotImplementedError

    def learnFromGame(self, states, decision):
        pass

    def startNewGame(self):
        pass

    def finishGame(self):
        pass

class RandomUTTTPlayer(UTTTPlayer):
    def makeNextMove(self, learn=True):
        previousState = self.board.getBoardState()
        if self.isBoardActive():
            nextBoardLocation = self.board.getNextBoardLocation()
            if None in nextBoardLocation:
                activeBoardLocations = self.board.getActiveBoardLocations()
                nextBoardLocation = random.choice(activeBoardLocations)
            emptyPlaces = self.board.getEmptyBoardPlaces(nextBoardLocation)
            pickOne = random.choice(emptyPlaces)

            self.board.makeMove(self.player, nextBoardLocation, pickOne)
        return previousState

    def learnFromMove(self, prevBoardState):
        pass  # Random player does not learn from move

class conv_RLUTTTPlayer(UTTTPlayer):
    def __init__(self, learningModel, gamma_exp = 0.1, gamma = 0.8):
        self.model = learningModel
        self.gamma_exp=gamma_exp
        self.gamma = gamma

    def testNextMove(self, state, boardLocation, placeOnBoard):
        loc = 27*boardLocation[0] + 9*boardLocation[1] + 3*placeOnBoard[0] + placeOnBoard[1]
        boardCopy = list(state)
        boardCopy[loc] = self.player
        return stateToNP(''.join(boardCopy))

    def makeNextMove(self, learn=True):
        previousState = self.board.getBoardState()

        if self.isBoardActive():
            nextBoardLocation = self.board.getNextBoardLocation()
            activeBoardLocations = [nextBoardLocation]
            if None in nextBoardLocation:
                activeBoardLocations = self.board.getActiveBoardLocations()
            
            moves = []
            next_states = []
            lp = []

            for boardLocation in activeBoardLocations:
                emptyPlaces = self.board.getEmptyBoardPlaces(boardLocation)
                for placeOnBoard in emptyPlaces:
                    possibleNextState = self.testNextMove(previousState, boardLocation, placeOnBoard)
                    moves.append((boardLocation, placeOnBoard))
                    next_states.append(possibleNextState)
                    
                    lpi = np.zeros([3,3])
                    lpi[placeOnBoard[0],placeOnBoard[1]] = 1
                    lp.append(lpi)


            v = self.model.predict([np.array(next_states)*self.playernum, np.array(lp)]).reshape([-1])
            p = np.exp(v)
            p = (1-self.gamma_exp)*(p/np.sum(p)) + self.gamma_exp*(np.ones_like(p)/p.size)
            
            if learn:
                q_chosen = np.random.choice(len(moves),1,p=p)[0]
            else:
                q_chosen = np.argmax(p)


            (chosenBoard, pickOne) = moves[q_chosen]

            self.board.makeMove(self.player, chosenBoard, pickOne)
        return previousState

    def learnFromGame(self, states, decision):
        moves = len(states)
        states = np.array([stateToNP(state) for state in states])
        if self.player == 'O':
            states = -1*states
        won = 0
        won_X = decision == UTTTBoardDecision.WON_X
        won_O = decision == UTTTBoardDecision.WON_O
        me_X = self.player == 'X'
        me_O = self.player == 'O'
        me_won = (me_X and won_X) or (me_O and won_O)
        me_lost = (me_X and won_O) or (me_O and won_X)
        if me_won:
            won = 1
        elif me_lost:
            won = -1
        labels = [won*(self.gamma**i) for i in range(moves)]
        labels.reverse()
        labels = np.array(labels)

        last_moves = states - np.roll(states,1,0)
        last_moves[0][:] = states[0][:]

        aux_data = []
        aux_rewards = []
        aux_lp = []

        for grid, reward in zip(states, labels):
            lpi = np.zeros([3,3],dtype='int')
            r,c = np.where(grid==1)
            lpi[r//3,c//3] = 1
            for i in range(4):
                temp = np.rot90(grid,i)
                temp_lpi = np.rot90(lpi,i)

                aux_data.append(temp)
                aux_lp.append(temp_lpi)
                aux_rewards.append(reward)
                aux_data.append(temp.T)
                aux_lp.append(temp_lpi.T)
                aux_rewards.append(reward)

        self.model.fit([np.array(aux_data, dtype='float32'),np.array(aux_lp)], np.array(aux_rewards, dtype='float32'), epochs=10, verbose=0)

class RLUTTTPlayer(UTTTPlayer):
    def __init__(self, learningModel):
        self.learningAlgo = learningModel
        super(RLUTTTPlayer, self).__init__()

    def printValues(self):
        self.learningAlgo.printValues()

    def testNextMove(self, state, boardLocation, placeOnBoard):
        loc = 27*boardLocation[0] + 9*boardLocation[1] + 3*placeOnBoard[0] + placeOnBoard[1]
        boardCopy = list(state)
        boardCopy[loc] = self.player
        return ''.join(boardCopy)

    def startNewGame(self):
        self.learningAlgo.resetForNewGame()

    def finishGame(self):
        self.learningAlgo.gameOver()

    def makeNextMove(self, learn=True):
        previousState = self.board.getBoardState()
        if self.isBoardActive():
            nextBoardLocation = self.board.getNextBoardLocation()
            activeBoardLocations = [nextBoardLocation]
            if None in nextBoardLocation:
                activeBoardLocations = self.board.getActiveBoardLocations()
            if random.uniform(0, 1) < 0.8:      # Make a random move with probability 0.2
                moveChoices = {}
                for boardLocation in activeBoardLocations:
                    emptyPlaces = self.board.getEmptyBoardPlaces(boardLocation)
                    for placeOnBoard in emptyPlaces:
                        possibleNextState = self.testNextMove(previousState, boardLocation, placeOnBoard)
                        moveChoices[(tuple(boardLocation), placeOnBoard)] = self.learningAlgo.getBoardStateValue(self.player, self.board, possibleNextState)
                (chosenBoard, pickOne) = max(moveChoices, key=moveChoices.get)
            else:
                chosenBoard = random.choice(activeBoardLocations)
                emptyPlaces = self.board.getEmptyBoardPlaces(chosenBoard)
                pickOne = random.choice(emptyPlaces)
            self.board.makeMove(self.player, chosenBoard, pickOne)
        return previousState

    def learnFromMove(self, prevBoardState):
        self.learningAlgo.learnFromMove(self.player, self.board, prevBoardState)

    def saveLearning(self, filename):
        self.learningAlgo.saveLearning(filename)

    def loadLearning(self, filename):
        self.learningAlgo.loadLearning(filename)

if __name__  == '__main__':
    board = UTTTBoard()
    player1 = RandomUTTTPlayer()
    player2 = RandomUTTTPlayer()
