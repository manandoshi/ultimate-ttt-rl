from board import GridStates, TTTBoardDecision
import json
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Reshape, LeakyReLU, Input, Conv2D, concatenate
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras.initializers import RandomNormal
import numpy as np

class GenericLearning(object):
    def getBoardStateValue(self, player, board, boardState):
        # Return the perceived `value` of a given board state
        raise NotImplementedError

    def learnFromMove(self, player, board, prevBoardState):
        # Learn from the previous board state and the current state of the board
        raise NotImplementedError

    def saveModel(self, filename):
        # Save to file (use pass if no implementation is necessary)
        # Useful for saving intermediate states of the learning model
        raise NotImplementedError

    def loadModel(self, filename):
        # Load an intermediate state of the learning model from file
        # Use only if also saving the intermediate state above
        raise NotImplementedError

    def resetForNewGame(self):
        pass

    def gameOver(self):
        pass

class TableLearning(GenericLearning):
    def __init__(self, DecisionClass=TTTBoardDecision):
        self.values = {}
        self.DecisionClass = DecisionClass

    def getBoardStateValue(self, player, board, boardState):
        decision = board.getBoardDecision()
        if decision == self.DecisionClass.WON_X:
            self.values[boardState] = 1.0 if player == GridStates.PLAYER_X else 0.0
        if decision == self.DecisionClass.WON_O:
            self.values[boardState] = 1.0 if player == GridStates.PLAYER_O else 0.0
        if decision == self.DecisionClass.DRAW or boardState not in self.values:
            self.values[boardState] = 0.5
        return self.values[boardState]

    def learnFromMove(self, player, board, prevBoardState):
        curBoardState = board.getBoardState()
        curBoardStateValue = self.getBoardStateValue(player, board, curBoardState)
        if prevBoardState not in self.values:
            self.getBoardStateValue(player, board, prevBoardState)
        self.values[prevBoardState] = self.values[prevBoardState] + 0.2*(curBoardStateValue - self.values[prevBoardState])

    def printValues(self):
        from pprint import pprint
        pprint(self.values)
        print('Total number of states: %s' % (len(self.values)))
        print('Total number of knowledgeable states: %s' % (len(filter(lambda x: x!=0.5, self.values.values()))))

    def saveLearning(self, filename):
        json.dump(self.values, open(filename,'w'))

    def loadLearning(self, filename):
        self.values = json.load(open(filename, 'r'))


class NNUltimateLearning(GenericLearning):
    STATE_TO_NUMBER_MAP = {GridStates.EMPTY: 0, GridStates.PLAYER_O: -1, GridStates.PLAYER_X: 1}

    def __init__(self, DecisionClass=TTTBoardDecision):
        self.DecisionClass = DecisionClass
        self.values = {}
        self.initializeModel()

    def initializeModel(self):
        self.model = Sequential()
        self.model.add(Dense(81, input_dim=81, activation='relu'))
        #self.model.add(Dense(81, activation='relu'))
        self.model.add(Dense(1, activation='linear', kernel_initializer='glorot_uniform'))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        plot_model(self.model, to_file='model.png')

    def initialModelTraining(self, jsonFile):
        # If the neural network model should be seeded from some known state/value pairs
        import os
        if os.path.isfile(jsonFile):
            self.values = json.load(open(jsonFile, 'r'))
            self.gameOver()

    def resetForNewGame(self):
        self.values = {}

    def gameOver(self):
        boardStates, predYs = [], []
        for (k,v) in self.values.iteritems():
            boardStates.append(self.convertBoardStateToInput(k))
            predYs.append(v)
        self.trainModel(boardStates, predYs)

    def convertBoardStateToInput(self, boardState):
        return list(map(lambda x: self.STATE_TO_NUMBER_MAP.get(x), boardState))

    def trainModel(self, boardStates, y):
        self.model.fit(np.asarray(boardStates), np.asarray(y), verbose=0)

    def getPrediction(self, boardState):
        return self.model.predict(np.asarray([self.convertBoardStateToInput(boardState)]))[0]

    def getBoardStateValue(self, player, board, boardState):  #TODO: Can batch the inputs to do several predictions at once
        decision = board.getBoardDecision()
        predY = self.getPrediction(boardState)[0]
        if decision == self.DecisionClass.WON_X:
            predY = 1.0 if player == GridStates.PLAYER_X else 0.0   #TODO: Explore using -1.0 instead of 0.0
            self.values[boardState] = predY
        if decision == self.DecisionClass.WON_O:
            predY = 1.0 if player == GridStates.PLAYER_O else 0.0
            self.values[boardState] = predY
        if decision == self.DecisionClass.DRAW:
            predY = 0.5
            self.values[boardState] = predY
        return predY

    def learnFromMove(self, player, board, prevBoardState):
        curBoardState = board.getBoardState()
        curBoardStateValue = self.getBoardStateValue(player, board, curBoardState)
        prevBoardStateValue = self.getPrediction(prevBoardState)[0]
        self.values[prevBoardState] = prevBoardStateValue + 0.2 * (curBoardStateValue - prevBoardStateValue)

    def printValues(self):
        pass

    def saveLearning(self, filename):
        self.model.save(filename)

    def loadLearning(self, filename):
        self.model = load_model(filename)


def generateModel():
    inp = Input(shape=(9,9,2))
    #r   = Reshape((9,9,1))(inp)
    r = inp

    x = Conv2D( filters = 5, kernel_size = (1,3), strides = (1,3), kernel_initializer='glorot_uniform' )(r)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D( filters = 5, kernel_size = (3,1), strides = (3,1), kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D( filters = 15, kernel_size = (3,3), kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Reshape((-1,15))(x)


    y = Conv2D( filters = 5, kernel_size = (3,1), strides = (3,1) , kernel_initializer='glorot_uniform')(r)
    y = LeakyReLU(alpha=0.1)(y)
    y = Conv2D( filters = 5, kernel_size = (1,3), strides = (1,3), kernel_initializer='glorot_uniform')(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = Conv2D( filters = 15, kernel_size = (3,3), kernel_initializer='glorot_uniform')(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = Reshape((-1,15))(y)


    z = Conv2D( filters = 100, kernel_size = (9,9), kernel_initializer='glorot_uniform')(r)
    z = LeakyReLU(alpha=0.1)(z)
    z = Reshape((-1,100))(z)

    f = concatenate([x,y,z])
    f = Dense(100)(f)
    f = LeakyReLU(alpha=0.1)(f)
    f = Dense(100)(f)
    f = LeakyReLU(alpha=0.1)(f)
    f = Dense(1, activation='tanh')(f)
    f = Reshape((-1,))(f)


    sgd = Adam(lr=0.01)

    model = Model(inputs=inp, outputs=f)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    return model
