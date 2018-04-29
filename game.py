from board import TTTBoardDecision, GridStates, TTTBoard
from ultimateboard import stateToNP

class SingleGame(object):
    def __init__(self, player1, player2, BoardClass=TTTBoard, BoardDecisionClass=TTTBoardDecision):
        self.player1 = player1
        self.player2 = player2
        self.board = BoardClass()
        self.BoardDecisionClass = BoardDecisionClass

    def playAGame(self):
        self.player1.startNewGame()
        self.player2.startNewGame()
        while self.board.getBoardDecision() == self.BoardDecisionClass.ACTIVE:
            self.player1.setBoard(self.board, GridStates.PLAYER_X)
            self.player2.setBoard(self.board, GridStates.PLAYER_O)
            pState1 = self.player1.makeNextMove()
            self.player1.learnFromMove(pState1)
            self.player2.learnFromMove(pState1)
            pState2 = self.player2.makeNextMove()
            self.player1.learnFromMove(pState2)
            self.player2.learnFromMove(pState2)
        self.player1.finishGame()
        self.player2.finishGame()
        return self.board.getBoardDecision()

    def playAGame2(self):
        self.player1.startNewGame()
        self.player2.startNewGame()
        self.player1.setBoard(self.board, GridStates.PLAYER_X)
        self.player2.setBoard(self.board, GridStates.PLAYER_O)
        while self.board.getBoardDecision() == self.BoardDecisionClass.ACTIVE:
            self.player1.makeNextMove()
            self.player2.makeNextMove()

        self.player1.finishGame()
        self.player2.finishGame()
        return self.board.getBoardDecision()

    def selfPlay(self):
        #player 1 will play with itself
        data = []
        while self.board.getBoardDecision() == self.BoardDecisionClass.ACTIVE:
            pState1 = player1.makeMove()



class GameSequence(object):
    def __init__(self, numberOfGames, player1, player2, BoardClass=TTTBoard, BoardDecisionClass=TTTBoardDecision):
        self.player1 = player1
        self.player2 = player2
        self.numberOfGames = numberOfGames
        self.BoardClass = BoardClass
        self.BoardDecisionClass = BoardDecisionClass

    def playGamesAndGetWinPercent(self):
        results = []
        for i in range(self.numberOfGames):
            game = SingleGame(self.player1, self.player2, self.BoardClass, self.BoardDecisionClass)
            results.append(game.playAGame())
        xpct, opct, drawpct = float(results.count(self.BoardDecisionClass.WON_X))/float(self.numberOfGames), \
                              float(results.count(self.BoardDecisionClass.WON_O))/float(self.numberOfGames), \
                              float(results.count(self.BoardDecisionClass.DRAW))/float(self.numberOfGames)
        return (xpct, opct, drawpct)

if __name__ == '__main__':
    from ultimateplayer import RandomUTTTPlayer, conv_RLUTTTPlayer
    from ultimateboard import UTTTBoard, UTTTBoardDecision
    from learning import generateModel
    model = generateModel()
    player1, player2 = conv_RLUTTTPlayer(model), conv_RLUTTTPlayer(model)
    game = SingleGame(player1, player2, UTTTBoard, UTTTBoardDecision)
    game.playAGame2()
