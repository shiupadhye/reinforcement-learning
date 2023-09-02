import gym
import itertools
import functools
import numpy as np
from gym import Env
from gym import spaces


BACKSPACE = "/b/"
INCLUDE_BACKSPACE = True


MAX_SEQ_LENGTH = 10


CORR_REWARD = 10
INCRM_PENALTY = -1
TIMEOUT_PENALITY = -10


class typeEnv(Env):
    def __init__(self):
        self.alphabet = sorted(functools.reduce(set.union, response_set, set()))
        self.actions = [a for a in self.alphabet]
        if INCLUDE_BACKSPACE:
            self.actions.append(BACKSPACE)
        self.states = []
        for i in range(MAX_SEQ_LENGTH):
            for combination in itertools.product(self.alphabet, repeat=i):
                state = "".join(combination)
                self.states.append(state)
        # construct action space
        self.action_space = spaces.Discrete(len(self.actions))
        # construct state space
        self.observation_space = spaces.Text(min_length=1,max_length=MAX_SEQ_LENGTH,charset=set(self.alphabet))
        # construct transition table
        self.state = self.states[0]
    
        
    def reset(self):
        self.state = self.states[0]
                

    def step(self,action):
        terminate = False
        # determine next state
        if action == "/b/":
            next_state = self.state[:-1]
        else:
            next_state = self.state + action
        ## determine reward
        if next_state in response_set:
            reward = CORR_REWARD
            terminate = True
        else:
            if len(next_state) >= MAX_SEQ_LENGTH:
                reward = TIMEOUT_PENALTY
            else:
                reward = INCRM_PENALTY
        
        self.state = next_state
        return (self.state,reward,terminate)
    
    def get_state(self):
        return self.state
    
    def get_actions(self):
        return self.actions
    
    def get_states(self):
        return self.states

