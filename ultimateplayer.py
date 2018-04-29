from ultimateboard import UTTTBoardDecision, UTTTBoard
from learning import TableLearning
import random
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Reshape, LeakyReLU, Input, Conv2D, concatenate
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras.initializers import RandomNormal

STATE_TO_NUMBER_MAP = {GridStates.EMPTY: 0, GridStates.PLAYER_O: -1, GridStates.PLAYER_X: 1}

class UTTTPlayer(object):
    def __init__(self):
        self.board = None
        self.player = None

    def setBoard(self, board, player):
        self.board = board
        self.player = player   # X or O

    def isBoardActive(self):
        return (self.board is not None and self.board.getBoardDecision() == UTTTBoardDecision.ACTIVE)

    def makeNextMove(self):
        raise NotImplementedError

    def learnFromMove(self, prevBoardState):
        raise NotImplementedError

    def startNewGame(self):
        pass

    def finishGame(self):
        pass

class RandomUTTTPlayer(UTTTPlayer):
    def makeNextMove(self):
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
    def __init__(self, learningModel):
        inp = Input(shape=(9,9,))
        r   = Reshape((9,9,1))(inp)

        x = Conv2D( filters = 5, kernel_size = (1,3), stride = (1,3) )(r)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D( filters = 5, kernel_size = (3,1), stride = (3,1) )(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D( filters = 15, kernel_size = (3,3))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Reshape((-1,15))(x)


        y = Conv2D( filters = 5, kernel_size = (3,1), stride = (3,1) )(r)
        y = LeakyReLU(alpha=0.1)(y)
        y = Conv2D( filters = 5, kernel_size = (1,3), stride = (1,3))(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Conv2D( filters = 15, kernel_size = (3,3))(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Reshape((-1,15))(y)


        z = Conv2D( filters = 20, kernel_size = (3,3), stride = (3,3) )(r)
        z = LeakyReLU(alpha=0.1)(z)
        z = Conv2D( filters = 15, kernel_size = (3,3))(z)
        z = LeakyReLU(alpha=0.1)(z)
        z = Reshape((-1,15))(z)

        f = concatenate([x,y,z])
        f = Dense(64)(f)
        f = LeakyReLU(alpha=0.1)(f)
        f = Dense(1, activation='tanh')(f)
        f = Reshape((-1,))(f)


        sgd = SGD(lr=0.001)

        self.model = Model(inputs=inp, outputs=f)
        self.model.compile(loss='mean_squared_error', optimizer=sgd)

    def testNextMove(self, state, boardLocation, placeOnBoard):
        loc = 27*boardLocation[0] + 9*boardLocation[1] + 3*placeOnBoard[0] + placeOnBoard[1]
        boardCopy = list(state)
        boardCopy[loc] = self.player
        return ''.join(boardCopy)

    def convertBoardStateToInput(self, boardState):
        return np.array(list(map(lambda x: STATE_TO_NUMBER_MAP.get(x), boardState))).reshape([9,9])

    def makeNextMove(self):
        previousState = self.board.getBoardState()
        if self.isBoardActive():
            nextBoardLocation = self.board.getNextBoardLocation()
            activeBoardLocations = [nextBoardLocation]
            if None in nextBoardLocation:
                activeBoardLocations = self.board.getActiveBoardLocations()
            
            moves = []
            next_states = []

            for boardLocation in activeBoardLocations:
                emptyPlaces = self.board.getEmptyBoardPlaces(boardLocation)
                for placeOnBoard in emptyPlaces:
                    possibleNextState = self.testNextMove(previousState, boardLocation, placeOnBoard)
                    moves.append((boardLocation, placeOnBoard))
                    next_states.append(self.convertBoardStateToInput(possibleNextState))

            gamma = 0
            v = self.model.predict(next_states)
            p = np.exp(v)
            p = (1-gamma_exp)*(p/np.sum(p)) + gamma_exp*(np.ones_like(p)/p.size)
            (chosenBoard, pickOne) = max(moveChoices, key=moveChoices.get)

            self.board.makeMove(self.player, chosenBoard, pickOne)
        return self.convertBoardStateToInput(previousState)


    
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

    def makeNextMove(self):
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
