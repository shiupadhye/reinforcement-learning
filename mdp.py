class MDP():
    def __init__(self,response_set,backspace,MAX_SEQ_LENGTH,TIMEOUT_LENGTH,CORR_REWARD,INCRM_PENALTY,TIMEOUT_PENALTY):
        # construct alphabet
        self.alphabet = sorted(functools.reduce(set.union, response_set, set()))
        # construct actions
        self.actions = [a for a in self.alphabet]
        self.actions.append(backspace)
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
    
    def reward_final(self,next_state_rep):
        terminate = False
        reward = -1
        if next_state_rep in response_set:
            if next_state_rep == response_set[0]:
                reward += CORR_REWARD
                terminate = True
            elif next_state_rep == response_set[1]:
                reward += CORR_REWARD
                terminate = True
            elif next_state_rep in response_set[2:]:
                reward += (-1 * CORR_REWARD)
                terminate = True
        else:
            if len(next_state_rep) == TIMEOUT_LENGTH:
                reward += TIMEOUT_PENALTY
                terminate = True
        return reward, terminate
        
    def transition(self,action):
        # get action rep
        action_rep = self.action_to_rep(action)
        # get state rep
        state_rep = self.state_to_rep(self.state)
        terminate = False
        if action_rep == "/b/":
            next_state_rep = state_rep[:-1]
        else:
            next_state_rep = state_rep + action_rep
            
        reward, terminate = self.reward_final(next_state_rep)
        next_state = self.rep_to_state(next_state_rep)
        self.state = next_state
        return (self.state,reward,terminate)