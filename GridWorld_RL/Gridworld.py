
# coding: utf-8

# In[94]:

import numpy as np

def randPair(s,e):
    return np.random.randint(s,e), np.random.randint(s,e)

#finds an array in the "depth" dimension of the grid
def findLoc(state, obj):
    for i in range(0,4):
        for j in range(0,4):
            if (state[i,j] == obj).all():
                return i,j

#Initialize stationary grid, all items are placed deterministically
def initGrid():
    state = np.zeros((4,4,5))
    #place player 1
    state[0,1] = np.array([0,0,0,1,0])
    #place wall
    state[2,2] = np.array([0,0,1,0,0])
    #place pit
    state[1,1] = np.array([0,1,0,0,0])
    #place goal
    state[3,3] = np.array([1,0,0,0,0])
    
    #place player 2
    state[randPair(0,4)] = np.array([0,0,0,0,1])
    a1 = findLoc(state, np.array([0,0,0,1,0])) #find grid position of player1 (agent)
    a2 = findLoc(state, np.array([0,0,0,0,1])) #find grid position of player2 
    w = findLoc(state, np.array([0,0,1,0,0])) #find wall
    g = findLoc(state, np.array([1,0,0,0,0])) #find goal
    p = findLoc(state, np.array([0,1,0,0,0])) #find pit
    if (not a1 or not a2 or not w or not g or not p):
        #print('Invalid grid. Rebuilding..')
        return initGrid()

    return state


# In[95]:

def makeMove(state, action,player2_terminated):
    #need to locate player in grid
    #need to determine what object (if any) is in the new grid spot the player is moving to
    player1_loc = getLoc(state, 3)
    if not player2_terminated:
        player2_loc = getLoc(state, 4)
    wall = findLoc(state, np.array([0,0,1,0,0]))
    goal = findLoc(state, np.array([1,0,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0,0]))
    state = np.zeros((4,4,5))

    #print player1_loc
    #print player2_loc
    #up (row - 1)
    if action==0:
        new_loc = (player1_loc[0] - 1, player1_loc[1])
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                state[new_loc][3] = 1
    #down (row + 1)
    elif action==1:
        new_loc = (player1_loc[0] + 1, player1_loc[1])
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                state[new_loc][3] = 1
    #left (column - 1)
    elif action==2:
        new_loc = (player1_loc[0], player1_loc[1] - 1)
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                state[new_loc][3] = 1
    #right (column + 1)
    elif action==3:
        new_loc = (player1_loc[0], player1_loc[1] + 1)
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                state[new_loc][3] = 1

    new_player1_loc = getLoc(state, 3)
    #print new_player1_loc
    if (not new_player1_loc):
        state[player1_loc] = np.array([0,0,0,1,0])
    #re-place pit
    state[pit][1] = 1
    #re-place wall
    state[wall][2] = 1
    #re-place goal
    state[goal][0] = 1
    if not player2_terminated:
        #re-place player 2
        state[player2_loc][4] = 1

    return state


# In[96]:

def makeMovePlayer2(state, player2_terminated, testing_mode, input_action = 0):
    #need to locate player in grid
    #need to determine what object (if any) is in the new grid spot the player is moving to
    player1_loc = getLoc(state, 3)
    player2_loc = getLoc(state, 4)
    wall = findLoc(state, np.array([0,0,1,0,0]))
    goal = findLoc(state, np.array([1,0,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0,0]))
    state = np.zeros((4,4,5))
    
    if testing_mode:
        #print player2_loc
        action = raw_input("Enter 0 for up, 1 for down, 2 for left, 3 for right ")
    elif not testing_mode:
        action = str(input_action)

    #up (row - 1)
    if action==str(0):
        new_loc = (player2_loc[0] - 1, player2_loc[1])
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                if new_loc != goal and new_loc != pit:
                    state[new_loc][4] = 1
                elif new_loc == goal:
                    state[new_loc] = np.array([1,0,0,0,1])
                elif new_loc == pit:
                        state[new_loc] = np.array([0,1,0,0,1])
    #down (row + 1)
    elif action==str(1):
        new_loc = (player2_loc[0] + 1, player2_loc[1])
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                if new_loc != goal and new_loc != pit:
                    state[new_loc][4] = 1
                elif new_loc == goal:
                    state[new_loc] = np.array([1,0,0,0,1])
                elif new_loc == pit:
                            state[new_loc] = np.array([0,1,0,0,1])
    #left (column - 1)
    elif action==str(2):
        new_loc = (player2_loc[0], player2_loc[1] - 1)
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                if new_loc != goal and new_loc != pit:
                    state[new_loc][4] = 1
                elif new_loc == goal:
                    state[new_loc] = np.array([1,0,0,0,1])
                elif new_loc == pit:
                            state[new_loc] = np.array([0,1,0,0,1])
    #right (column + 1)
    elif action==str(3):
        new_loc = (player2_loc[0], player2_loc[1] + 1)
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                if new_loc != goal and new_loc != pit:
                    state[new_loc][4] = 1
                elif new_loc == goal:
                    state[new_loc] = np.array([1,0,0,0,1])
                elif new_loc == pit:
                            state[new_loc] = np.array([0,1,0,0,1])

    new_player2_loc = getLoc(state, 4)
    if (not new_player2_loc):
        state[player2_loc] = np.array([0,0,0,0,1])
    #re-place pit
    state[pit][1] = 1
    #re-place wall
    state[wall][2] = 1
    #re-place goal
    state[goal][0] = 1
     #re-place player 1
    state[player1_loc][3] = 1

    return state


# In[97]:

def getLoc(state, level):
    for i in range(0,4):
        for j in range(0,4):
            if (state[i,j][level] == 1):
                return i,j

def getReward(state):
    player_loc = getLoc(state, 3)
    pit = getLoc(state, 1)
    goal = getLoc(state, 0)
    if (player_loc == pit):
        return -10
    elif (player_loc == goal):
        return 10
    else:
        return -1
    
def getRewardPlayer2(state):
    player2_loc = getLoc(state, 4)
    pit = getLoc(state, 1)
    goal = getLoc(state, 0)
    if (player2_loc == pit):
        return -10
    elif (player2_loc == goal):
        return 10
    else:
        return -1

def dispGrid(state,player2_terminated):
    grid = np.zeros((4,4), dtype='<U2')
    player1_loc = getLoc(state, 3)
    player2_loc = getLoc(state, 4)
    wall = findLoc(state, np.array([0,0,1,0,0]))
    goal = findLoc(state, np.array([1,0,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0,0]))
    for i in range(0,4):
        for j in range(0,4):
            grid[i,j] = ' '

    if player1_loc:
        grid[player1_loc] = 'P1' #player1
    if player2_loc:
         if not player2_terminated:
            grid[player2_loc] = 'P2' #player2
    if player1_loc == player2_loc and not player2_terminated:
        grid[player1_loc] = 'PB' #player1 and player2
    if wall:
        grid[wall] = 'W' #wall
    if goal:
        grid[goal] = '+' #goal
    if pit:
        grid[pit] = '-' #pit

    return grid


# In[98]:

state = initGrid()
dispGrid(state, True)


# In[99]:

state = makeMove(state, 0,True)
print('Reward: %s' % (getReward(state),))
dispGrid(state,True)


# In[100]:

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.models import model_from_json
import os


# In[101]:

model = Sequential()
model.add(Dense(164, init='lecun_uniform', input_shape=(80,)))
model.add(Activation('relu'))
#model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

model.add(Dense(150, init='lecun_uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Dense(4, init='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)


# In[102]:

model.predict(state.reshape(1,80), batch_size=1)
#just to show an example output; read outputs left to right: up/down/left/right


# In[103]:

from IPython.display import clear_output
import random

player2_terminated = False
epochs = 1000
gamma = 0.9 #since it may take several moves to goal, making gamma high
epsilon = 1
for i in range(epochs):
    state = initGrid()
    status = 1
    player2_terminated = False
    #while game still in progress
    while(status == 1):
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state.reshape(1,80), batch_size=1)
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,4)
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state = makeMove(state, action,player2_terminated)
        if not player2_terminated and getReward(new_state) == -1:
            new_state = makeMovePlayer2(new_state,player2_terminated,False,action)
            if getRewardPlayer2(new_state) == -10:
                #place pit
                new_state[1,1] = np.array([0,1,0,0,0])
                player2_terminated = True
            elif getRewardPlayer2(new_state) == 10:
                #place goal
                new_state[3,3] = np.array([1,0,0,0,0])
                player2_terminated = True
        #Observe reward
        reward = getReward(new_state)
        #Get max_Q(S',a)
        newQ = model.predict(new_state.reshape(1,80), batch_size=1)
        maxQ = np.max(newQ)
        y = np.zeros((1,4))
        y[:] = qval[:]
        if reward == -1: #non-terminal state
            update = (reward + (gamma * maxQ))
        else: #terminal state
            update = reward
        y[0][action] = update #target output
        print("Game #: %s" % (i,))
        model.fit(state.reshape(1,80), y, batch_size=1, nb_epoch=10, verbose=1)
        state = new_state
        if reward != -1:
            status = 0
        clear_output(wait=True)
    if i== 20 or i == 50 or i == 70 or i == 100 or i == 400 or i == 700 or i == 999:
        # serialize model to JSON
        file_name = "model"+str(i)
        model_json = model.to_json()
        with open(file_name+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(file_name+".h5")
        #print("Saved model to disk")
    
    if epsilon > 0.1:
        epsilon -= (1/epochs)


# In[104]:

def testAlgo(testModel,init=0):
    i = 0
    state = initGrid()

    player2_terminated = False
    state1 = state
    print("Initial State:")
    print(dispGrid(state,player2_terminated))
    status = 1
    #while game still in progress
    while(status == 1):
        qval = testModel.predict(state1.reshape(1,80), batch_size=1)
        action = (np.argmax(qval)) #take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action))
        state1 = makeMove(state1, action,player2_terminated)
        print(dispGrid(state1,player2_terminated))
        reward = getReward(state1)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
            break
        if not player2_terminated:
            state2 = makeMovePlayer2(state1,player2_terminated,True)
            state1 = state2
            if getRewardPlayer2(state1) == -10:
                #place pit
                state1[1,1] = np.array([0,1,0,0,0])
                player2_terminated = True
            elif getRewardPlayer2(state1) == 10:
                #place goal
                state1[3,3] = np.array([1,0,0,0,0])
                player2_terminated = True
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 10):
            print("Game lost; too many moves.")
            break


# In[105]:

#testAlgo(model,init=0)


# In[108]:

def test_after_epochs():
    count = [20,50,70,100,400,700,999]
    #count = [999]
    for i in count:
        file_to_load = "model"+str(i)
        # load json and create model
        json_file = open(file_to_load+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        testModel = model_from_json(loaded_model_json)
        # load weights into new model
        testModel.load_weights(file_to_load+".h5")
        testModel.compile(loss='mse', optimizer=rms)
        print("Loaded model at epoch "+str(i)+" from disk")
        testAlgo(testModel,init=0)


# In[109]:

test_after_epochs()





