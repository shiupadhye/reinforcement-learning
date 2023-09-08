class seqMDP():
    def __init__(self,response_set,goals,goal_probs,backspace,MAX_SEQ_LENGTH,TIMEOUT_LENGTH,CORR_REWARD,INCRM_PENALTY,TIMEOUT_PENALTY):
        # construct alphabet
        self.alphabet = sorted(functools.reduce(set.union, response_set, set()))
        # construct action space
        self.actions = [a for a in self.alphabet]
        self.actions.append(backspace)
        self.n_actions = len(self.actions)
        self.action_space = dict(zip(np.arange(self.n_actions),self.actions))
        # construct goal space
        self.goals = goals
        self.n_goals = len(self.goals)
        self.goal_probs = goal_probs
        self.goal_space = dict(zip(np.arange(self.n_goals),self.goals))
        # construct observation space
        self.start = ''
        alphabetwithStart = [self.start] + self.alphabet
        self.states = []
        self.goal_state_pairs = []
        for i in range(MAX_SEQ_LENGTH):
            for subseq in itertools.product(alphabetwithStart, repeat=i):
                state = "".join(subseq)
                if state not in self.states:
                    self.states.append(state)
                    for goal in goals:
                        self.goal_state_pairs.append((goal,state))
        self.n_states = len(self.states)
        self.state_space = dict(zip(np.arange(self.n_states),self.states))
        self.n_observations = len(self.goal_state_pairs)
        self.observation_space = dict(zip(self.goal_state_pairs,np.arange(len(self.goal_state_pairs))))
        
    def get_state(self):
        return self.state
    
    def get_goal(self):
        return self.goal
    
    def get_goal_state_idx(self,goal_state):
        return self.observation_space[goal_state]
    
    def state_to_rep(self,state):
        return self.state_space[state]
    
    def action_to_rep(self,action):
        return self.action_space[action]
    
    def goal_to_rep(self,goal):
        return self.goal_space[goal]
        
    def rep_to_state(self,state_rep):
        rep_to_state = {v: k for k, v in self.state_space.items()}
        return rep_to_state[state_rep]
    
    def rep_to_action(self,action_rep):
        rep_to_action = {v: k for k, v in self.action_space.items()}
        return rep_to_action[action_rep]
    
    def reset(self):
        self.state = self.start
        self.goal = np.random.choice(self.goals,1,self.goal_probs)[0]
    
    def sample_action(self):
        probs = [1/self.n_actions for i in range(self.n_actions)]
        return np.random.choice(np.arange(self.n_actions),1,probs)[0]
    
    def reward_final(self,next_state):
        terminate = False
        reward = -1
        if next_state in response_set:
            if next_state == response_set[0] and self.goal == "name":
                reward += CORR_REWARD
                terminate = True
            elif next_state == response_set[0] and self.goal == "read":
                reward += CORR_REWARD/2
                terminate = True
            elif next_state == response_set[1] and self.goal == "read":
                reward += CORR_REWARD
                terminate = True
            elif next_state == response_set[1] and self.goal == "name":
                reward += CORR_REWARD/2
                terminate = True
            else:
                reward += (-1 * CORR_REWARD)
                terminate = True
        else:
            if len(next_state) == TIMEOUT_LENGTH:
                reward += TIMEOUT_PENALTY
                terminate = True
        return reward, terminate
        
    def transition(self,action):
        # get action rep
        action_rep = self.action_to_rep(action)
        # get state rep
        curr_state = self.state
        terminate = False
        # if backspace, return to prev state
        if action_rep == "/b/":
            next_state = curr_state[:-1]
        # if not, increment
        else:
            next_state = curr_state + action_rep
        reward, terminate = self.reward_final(next_state)
        self.state = next_state
        return (self.state,reward,terminate)