import time
import itertools
import functools
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt


BACKSPACE = "/b/"
INCLUDE_BACKSPACE = True
MAX_SEQ_LENGTH = 10
TIMEOUT_LENGTH = 5
CORR_REWARD = 10
INCRM_PENALTY = -1
TIMEOUT_PENALTY = -50

response_set = ["dog","cat"]

class MDP():
    def __init__(self):
        # construct alphabet
        self.alphabet = sorted(functools.reduce(set.union, response_set, set()))
        # construct actions
        self.actions = [a for a in self.alphabet]
        if INCLUDE_BACKSPACE:
            self.actions.append(BACKSPACE)
        # construct action space
        self.n_actions = len(self.actions)
        self.action_map = dict(zip(np.arange(self.n_actions),self.actions))
        # construct state space
        self.start = ''
        alphabetwithStart = [self.start] + self.alphabet
        self.states = []
        for i in range(MAX_SEQ_LENGTH):
            for combination in itertools.product(alphabetwithStart, repeat=i):
                state = "".join(combination)
                self.states.append(state)
        self.n_states = len(self.states)
        self.state_map = dict(zip(np.arange(self.n_states),self.states))
        
        
    def get_state(self):
        return self.state
    
    def state_to_rep(self,state):
        return self.state_map[state]
    
    def action_to_rep(self,action):
        return self.action_map[action]
        
    def rep_to_state(self,state_rep):
        rep_to_state = {v: k for k, v in self.state_map.items()}
        return rep_to_state[state_rep]
    
    def rep_to_action(self,action_rep):
        rep_to_action = {v: k for k, v in self.action_map.items()}
        return rep_to_action[action_rep]
    
    def reset(self):
        self.state = self.rep_to_state(self.start)

    def sample_action(self):
        probs = [1/self.n_actions for i in range(self.n_actions)]
        return np.random.choice(np.arange(self.n_actions),1,probs)[0]
    
    def step(self,action):
        # get action rep
        action_rep = self.action_to_rep(action)
        # get state rep
        state_rep = self.state_to_rep(self.state)
        terminate = False
        if action_rep == "/b/":
            next_state_rep = state_rep[:-1]
        else:
            next_state_rep = state_rep + action_rep
        ## determine reward
        if next_state_rep in response_set:
            reward = CORR_REWARD
            terminate = True
        else:
            if len(next_state_rep) == TIMEOUT_LENGTH:
                reward = TIMEOUT_PENALTY
                terminate = True
            else:
                reward = INCRM_PENALTY
        next_state = self.rep_to_state(next_state_rep)
        self.state = next_state
        return (self.state,reward,terminate)


# implementation of epsilon greedy action selection (multi-armed bandit)
def epsilon_greedy_agent(epsilon,state,QTable):
    x = bernoulli.rvs(1-epsilon, size=1)
    if x == 1:
        # exploit: select best performing action w/ probability 1 - epsilon
        action = np.argmax(QTable[state])
    else:
        # explore: randomly select action for exploration w/ probability epsilon
        action = env.sample_action()
    return action

def TD_error(reward,gamma,QTable,next_state,state,action):
    delT = reward + gamma * np.max(QTable[next_state]) - QTable[state,action]
    return delT

# From Sutton and Barto (2018), p. 131
def train_Qlearning(alpha,epsilon,gamma,episodes,maxSteps,verbose=False):
    print("Begin training...")
    # initialize Q-table
    QTable = np.zeros([env.n_states,env.n_actions])
    stepsPerEp = []
    for ep in range(episodes):
        if verbose and ep % 1000 == 0:
            print("Episode %d" % ep)
        # initialize state
        env.reset()
        state = env.get_state()
        isDone = False
        numSteps = 0
        for t in range(maxSteps):
            # sample action from Q
            action = epsilon_greedy_agent(epsilon,state,QTable)
            # take action, observe reward and next state
            next_state,reward,terminate = env.step(action)
            # update Q and state
            delT = TD_error(reward,gamma,QTable,next_state,state,action)
            QTable[state,action] += alpha * delT
            state = next_state
            # until state is terminal
            if terminate:
                break
            numSteps += 1
        stepsPerEp.append(numSteps)
    return QTable,stepsPerEp  


env = MDP()

episodes = 50000
maxSteps = 10000
alpha = 0.1
epsilon = 0.1
gamma = 0.1
lam = 0.1
QTable,stepsPerEp = train_Qlearning(alpha,epsilon,gamma,episodes,maxSteps,verbose=True)


stepsPerEp = np.array(stepsPerEp)
plt.plot(np.arange(episodes)+1,stepsPerEp)


testEpisodes = 10
epRewards = {}
epPenalities = {}
epTimesteps = {}
epTermStates = {}
verbose = True
for ep in range(testEpisodes):
    if verbose:
        print("Episode %d" % (ep+1))
    start_time = time.time()
    env.reset()
    state = env.get_state()
    rewards = []
    isDone = False
    rewards = 0
    timesteps = 0
    penalities = 0
    states = []
    while not isDone:
        action = np.argmax(QTable[state,:])
        next_state,reward,isDone = env.step(action)
        state = next_state
        timesteps += 1
        if reward == -10:
            penalities += 1
        rewards += reward
        states.append(env.state_to_rep(state))
    if verbose:
        print("--- Completed in %s seconds ---" % (time.time() - start_time))
    epTermStates[ep] = states
    epRewards[ep] = rewards
    epPenalities[ep] = penalities
    epTimesteps[ep] = timesteps


