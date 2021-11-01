import numpy as np
#import random
import copy

from numpy.core.numeric import moveaxis

KEY_TIE = 'tie'
KEY_ONGOING = 'ongoing'
KEY_WON_P1 = 'won_p1'
KEY_WON_P2 = 'won_p2'
KEY_P1 = 1
KEY_P2 = 2

alpha = 0.1
gamma = 1
epsilon = 2 # could be set decaying after every 100 games, start at 1

win_reward = 1
lose_reward = -1
tie_reward = 0

n_games = 50#30000

def update_state(state, _q_table, epsilon, player_id):

    q_values = get_q_values(_q_table, state)
    
    r = np.random.rand()
    
    if np.all(np.isnan(q_values)) and not np.count_nonzero(state==0) == 0:
        print('ERROR: q values all zero but still a free spot')
        print(_q_table)

    #possible_actions = np.array([(idx, value) for (idx, value) in enumerate(q_values.flatten()) if not np.isnan(value) ])
    if np.all(np.isnan(q_values)): #len(possible_actions) < 1:
        #print('ERROR: all fields already filled, game ended')
        return state

    if r < (1-epsilon):
        # find max of q table
        action_idx = np.unravel_index(np.argmax(q_values, axis=None), q_values.shape)
    else:
        non_nan_idxs = np.argwhere(~np.isnan(q_values))
        r_idx = np.random.randint(non_nan_idxs.shape[0])
        action_idx = non_nan_idxs[r_idx,:]

    new_state = copy.copy(state)
    new_state[action_idx[0], action_idx[1]] = player_id

    return new_state

def game_status(state):
    # check if tie
    print(np.count_nonzero(state==0))
    if np.count_nonzero(state==0) == 0:
        return KEY_TIE
    # check if someone has won
    possible_winning_comb = np.zeros((8, 3))
    possible_winning_comb[:3,:] = state
    possible_winning_comb[3:6,:] = state.T
    possible_winning_comb[6,:] = state.diagonal()
    possible_winning_comb[7,:] = np.fliplr(state).diagonal()
    #print('check')
    for row in possible_winning_comb:
        if np.all(row == KEY_P1):
            return KEY_WON_P1
        elif np.all(row == KEY_P2):
            return KEY_WON_P2

    return KEY_ONGOING

def get_q_values(_q_table, state):

    state_idx = [i for (i, val) in enumerate(_q_table['states']) if (val==state).all()]

    if len(state_idx) == 1:
        q_values = copy.copy(_q_table['q_values'][state_idx[0]])
    elif len(state_idx) == 0:
        q_values = np.zeros([3,3]) 
    else:
        print('ERROR: state index in get_q_value has unexpeted length of ', len(state_idx))

    it = np.nditer(state, flags=['multi_index'])
    for x in it:
        if x != 0:
            q_values[it.multi_index] = np.nan

    return q_values

def update_q_table(_q_table, state, reward, player_id):

    current_q_values = get_q_values(_q_table, state)
    it = np.nditer(state, flags=['multi_index'])
    for x in it:

        if x == 0:
            next_state = copy.copy(state)
            next_state[it.multi_index] = player_id

            next_q_values = get_q_values(_q_table, next_state)

            max_q_values = np.nanmax(next_q_values)
            if max_q_values == np.nan:
                max_q_values = 0

            current_q_values[it.multi_index] += alpha*(reward + gamma * max_q_values - current_q_values[it.multi_index])
        else:
            current_q_values[it.multi_index] = np.nan

    #state_idx = [i for (i, val) in enumerate(_q_table['states']) if np.array_equal(val,state)]
    state_idx = [i for (i, val) in enumerate(_q_table['states']) if (val==state).all()]
    if len(state_idx) == 1:
        _q_table['q_values'][state_idx[0]] = current_q_values
    elif len(state_idx) == 0:
        _q_table['states'].append(state)
        _q_table['q_values'].append(current_q_values)
    else:
        print('ERROR: wrong length of state idx')
    return _q_table


if __name__ == '__main__':
    q_table1 = {'states': [],
                'q_values' :[]}
                
    q_table2 = {'states': [],
                'q_values' :[]}

    for i in range(n_games):

        print('game ', i)
        # init board
        result = None
        reward = 0
        board = np.zeros([3,3])

        #print('Board: ', board)
        # player 1 moves
        board = update_state(board, q_table1, epsilon, KEY_P1)
        # player 2 moves
        board = update_state(board, q_table2, epsilon, KEY_P2)
        #print('Board: ', board)

        # update q1
        q_table1 = update_q_table(q_table1, board, reward, KEY_P1)

        while game_status(board) == KEY_ONGOING:
            # player 1 move 

            board = update_state(board, q_table1, epsilon, KEY_P1)

            _status = game_status(board)
            if _status != KEY_ONGOING:
                result = _status
                break
            else:
                # update Q2
                q_table2 = update_q_table(q_table2, board, reward, KEY_P2)


            # player 2 move 
            board = update_state(board, q_table2, epsilon, KEY_P2)

            _status = game_status(board)
            if _status != KEY_ONGOING:
                result = _status
                break
            else:
                # update Q1
                q_table1 = update_q_table(q_table1, board, reward, KEY_P1)

        print('Board: ', board)
        print('Result: ', result)

        # update q1 and q2 with corresponding reward, reward only on last position of players
        if _status == KEY_WON_P1:
            
            q_table1 = update_q_table(q_table1, board, win_reward, KEY_P1)
            q_table2 = update_q_table(q_table2, board, lose_reward, KEY_P2)
        elif _status == KEY_WON_P2:
            q_table1 = update_q_table(q_table1, board, lose_reward, KEY_P1)
            q_table2 = update_q_table(q_table2, board, win_reward, KEY_P2)  
        elif _status == KEY_TIE:       
            q_table1 = update_q_table(q_table1, board, tie_reward, KEY_P1)
            q_table2 = update_q_table(q_table2, board, tie_reward, KEY_P2)  
        else:
            print('ERROR: unexpected result')


print('qt 1: ', q_table1)
print('qt 2: ', q_table2)