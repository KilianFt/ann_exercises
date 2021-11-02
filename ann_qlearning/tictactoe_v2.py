import numpy as np

class Board:
    def __init__(self, player1, player2):
        self.board = np.zeros([3,3])
        self.player1 = player1
        self.player2 = player2
        self.winner = None

        self.win_reward = 1
        self.lose_reward = -1
        self.tie_reward = 0

        self.current_player = player1.id

        self.game_finished = False

    def get_possible_actions(self):
        possible_idxs = np.argwhere(self.board==0)
        return possible_idxs

    def update_state(self, action):
        self.board[action] = self.current_player

        self.current_player = self.current_player - (2*self.current_player)

    def reset_board(self):
        self.board = np.zeros([3,3])
        self.current_player = self.player1.id
        self.game_finished = False

    def update_rewards(self):
        if self.winner == self.player1:
            self.player1.give_reward(self.win_reward)
            self.player2.give_reward(self.lose_reward)
        elif self.winner == self.player2:
            self.player1.give_reward(self.lose_reward)
            self.player2.give_reward(self.win_reward)
        else:
            self.player1.give_reward(self.tie_reward)
            self.player2.give_reward(self.tie_reward)  

    def game_state(self):
        # check if someone has won
        possible_winning_comb = np.zeros((8, 3))
        possible_winning_comb[:3,:] = self.board
        possible_winning_comb[3:6,:] = self.board.T
        possible_winning_comb[6,:] = self.board.diagonal()
        possible_winning_comb[7,:] = np.fliplr(self.board).diagonal()
        #print('check')
        for row in possible_winning_comb:
            if np.all(row == self.player1.id):
                self.winner = self.player1.id
                return self.player1.id
            elif np.all(row == self.player2.id):
                self.winner = self.player2.id
                return self.player2.id

        # check if tie
        if np.count_nonzero(self.board==0) == 0:
            self.winner = 100
            return 100

        return None

    def get_board_string(self):
        return str(self.board.reshape(9))

    def play_game(self, n_games):
        for i in range(n_games):
            print('game ', i)

            while not self.game_finished:

                possible_actions = self.get_possible_actions()
                player1_action = self.player1.choose_action(possible_actions, self.board, self.current_player)

                self.update_state(player1_action)

                board_string = self.get_board_string()
                self.player1.add_board(board_string)

                status = self.game_state()

                if status is not None:
                    self.update_rewards()
                    self.player1.reset()
                    self.player2.reset()
                    self.reset_board()
                    break
                
                possible_actions = self.get_possible_actions()
                player2_action = self.player2.choose_action(possible_actions, self.board, self.current_player)

                self.update_state(player2_action)

                board_string = self.get_board_string()
                self.player2.add_board(board_string)

                status = self.game_state()

                if status is not None:
                    self.update_rewards()
                    self.player1.reset()
                    self.player2.reset()
                    self.reset()
                    break          

class Player:
    def __init__(self, id, alpha, epsilon, gamma):
        self.id = id
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.q_table = {}

        self.boards = []

    def get_board_string(self):
        return str(self.board.reshape(9))

    def choose_action(self, actions, board, player_id):
        r = np.random.rand()
        if r < (1 - self.epsilon):
            value_max = -999
            for p in actions:
                next_board = board.copy()
                next_board[p] = player_id
                next_board_string = self.getHash(next_board)
                value = 0 if self.q_table.get(next_board_string) is None else self.q_table.get(next_board_string)
                # print("value", value)
                
                if value >= value_max:
                    value_max = value
                    action = p
        else:
            r_idx = np.random.choice(len(actions))
            action = actions[r_idx]

        return action

    def add_board(self, board):
        self.boards.append(board)

    def give_reward(self, reward):
        for st in reversed(self.boards):
            if self.q_table.get(st) is None:
                self.q_table[st] = 0
            self.q_table[st] += self.alpha * (self.gamma * reward - self.q_table[st])
            reward = self.q_table[st]

    def reset(self):
        self.boards = []


if __name__ == "__main__":
    alpha = 0.1
    gamma = 1
    epsilon = .2 # check value
    
    player1 = Player(1, alpha, gamma, epsilon)
    player2 = Player(-1, alpha, gamma, epsilon)

    game = Board(player1, player2)
    game.play_game(5)