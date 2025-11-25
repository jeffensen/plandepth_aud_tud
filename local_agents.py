#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.distributions import Categorical, Bernoulli
from numpy import nan

zeros = torch.zeros
ones = torch.ones
randn = torch.randn


class BackInduction_full_planning_with_alpha(object):
    def __init__(self,
                 planet_confs, # Matrix of zeros and ones
                 runs=1, # number of parallel runs (i.e. agents). For each run, one can specify a different set of model parameters.
                 mini_blocks=1,
                 trials=1,
                 na=2, # no of actions
                 ns=6, # no of states
                 costs=None,
                 utility=None, # Utility of planet types. Can be set to rewards of the planets
                 planning_depth=1,
                 depths=None,
                 variable_depth=False):
        
        self.runs = runs
        self.nmb = mini_blocks
        self.trials = trials
        self.np = 3  # number of free model parameters

        self.depth = planning_depth  # maximal planning depth
        if depths is None:
            self.depths = [torch.tensor([planning_depth - 1]).repeat(self.runs)]
        if variable_depth:
            self.make_depth_transitions(rho=.8)
        else:
            self.make_depth_transitions()
        self.na = na  # number of actions
        self.ns = ns  # number of states

        # matrix containing planet type in each state
        self.pc = planet_confs
        
        if costs is not None:
            self.costs = costs
        else:
            self.costs = torch.tensor([-.2, -.5])

        if utility is not None:
            self.utility = utility
        else:
            self.utility = torch.arange(-2., 3., 1.)

        self.transitions = torch.tensor([4, 3, 4, 5, 1, 1])

    def set_parameters(self, trans_par=None, true_params=False):
        # [INPUT]
        # 1) None OR
        # 2) torch.tensor([a, b, c]), where a = ln(beta), b = theta, and c = logit(alpha) OR
        # 3) torch.tensor([a, b, c], true_params=True), where a = beta, b = theta, and c = alpha

        if trans_par is not None:
            if true_params:
                self.beta = (trans_par[..., 0]) # Response noise
                self.theta = trans_par[..., 1] # Bias term in sigmoid function fot action-selection
                self.alpha = (trans_par[..., 2]) # Learning rate for belief update     
            else:
                assert trans_par.shape[-1] == self.np
                self.beta = (trans_par[..., 0]).exp() # Response noise
                self.theta = trans_par[..., 1] # Bias term in sigmoid function fot action-selection
                self.alpha = (trans_par[..., 2]).sigmoid() # Learning rate for belief update                

        else:
            self.beta = torch.tensor([10.]).repeat(self.runs)
            self.theta = zeros(self.runs)
            self.alpha = zeros(self.runs)

        self.batch_shape = self.beta.shape

        self.tp_mean0 = torch.tensor([.9, .5]).expand(self.batch_shape + (2,))

        self.tau = torch.tensor(1e10).expand(self.batch_shape)

        # (Estimated) probability of successful jump
        self.tp_mean = [self.tp_mean0]

        # state transition matrices
        self.tm = []

        # expected state value
        self.Vs = []
        
        # expected action values
        self.Q = []

        # action value difference: Q(a=jump) - Q(a=move)
        self.D = []

        # response probability
        self.logits = []

    def make_depth_transitions(self, rho=1.):

        tm = torch.eye(self.depth).repeat(self.runs, 1, 1)
        if self.depth > 1:
            tm = rho*tm + (1-rho)*(ones(self.depth, self.depth) - tm)/(self.depth-1)

        self.tm_depths = tm

    def make_transition_matrix(self, p):
        # INPUT: p = probability of successful jump
        # Give p as tensor with #runs entries.
        na = self.na  # number of actions
        ns = self.ns  # number of states
        shape = self.batch_shape  # number of runs

        tm = zeros(shape + (na, ns, ns))

        # move left action - no tranistion uncertainty
        tm[..., 0, :-1, 1:] = torch.eye(ns-1)
        tm[..., 0, -1, 0] = 1

        # jump action - with varying levels of transition uncertainty
        tm[..., 1, -2:, 0:3] = (1 - p.reshape(shape+(1, 1)).expand(shape + (2, 3)))/2
        tm[..., 1, -2:, 1] = p.reshape(shape + (1,)).expand(shape + (2,))

        z = (1 - p.reshape(shape + (1,))).expand(shape + (3,))/2
        tm[..., 1, 2, 3:6] = z
        tm[..., 1, 0, 3:6] = z
        tm[..., 1, 1, 2:5] = z

        tm[..., 1, 2, 4] = p
        tm[..., 1, 0, 4] = p
        tm[..., 1, 3, 0] = (1 - p)/2
        tm[..., 1, 3, -2] = (1 - p)/2
        tm[..., 1, 3, -1] = p
        tm[..., 1, 1, 3] = p

        self.tm.append(tm)

    def compute_state_values(self, block):

        tm = self.tm[-1]  # transition matrix
        depth = self.depth  # planning depth
        shape = self.batch_shape

        utility = self.utility

        Vs = [torch.sum(utility * self.pc[:, block], -1).expand(shape+(self.ns,))]
        
        # action value difference: Q(a=jump) - Q(a=move)
        D = []

        R = torch.einsum('...ijkl,...il->...ikj', tm, Vs[-1]) + self.costs

        Q = R
        
        Qs = []
        
        for d in range(1, depth+1):
            
            Qs.append(Q)
            
            # compute Q value differences for different actions
            dQ = Q[..., 1] - Q[..., 0]

            # compute response probability
            p = (dQ * self.tau[..., None]).sigmoid()

            # set state value (=expected reward if one would start from this planet depending on depth iteration)
            Vs.append(p * Q[..., 1] + (1-p) * Q[..., 0])

            D.append(dQ)

            if d < depth:
                Q = torch.einsum('...ijkl,...il->...ikj', tm, Vs[-1]) + R

        self.Vs.append(torch.stack(Vs))
        self.D.append(torch.stack(D, -1))
        
        self.Q.append(torch.stack(Qs))

    def update_beliefs(self, block, trial, states, conditions, responses=None):
        # conditions is a (2 x #runs x #miniblocks) tensor
        # conditions[0, :, :] is noise condition (0: lownoise, 1: highnoise)
        # conditions[1, :, :] is no of steps in miniblocks
        self.noise = conditions[0]
        self.max_trials = conditions[1]

        subs = torch.arange(self.runs)
        alpha = self.alpha

        if trial == 0:
            # update_transition_probability
            self.make_transition_matrix(self.tp_mean[-1][..., subs, self.noise])

        else:
            # update beliefs update state transitions
            if trial == 3:
                lr = alpha * (responses > 0).float()
            else:
                lr = responses * alpha
            succesful_transitions = (self.transitions[self.states] == states).float()
            probs = self.tp_mean[-1][..., subs, self.noise]

            probs_new = probs + lr * (succesful_transitions - probs)

            tp_mean = self.tp_mean[-1].clone()

            tp_mean[..., subs, self.noise] = probs_new

            self.tp_mean.append(tp_mean)

            self.make_transition_matrix(probs_new)

        # set beliefs about state (i.e. location of agent) to observed states
        self.states = states

    def plan_actions(self, block, trial):

        self.compute_state_values(block)

        D = self.D[-1][..., range(self.runs), self.states, :]

        beta = self.beta[..., None]
        theta = self.theta[..., None]
        self.logits.append(D * beta + theta)
        
    def sample_responses(self, block, trial):
        if trial == 0 and block > 0:
            probs = self.tm_depths[range(self.runs), self.depths[0]]
            depths = Categorical(probs=probs).sample()
            loc = depths > self.max_trials - 1
            depths[loc] = self.max_trials[loc] - 1
            self.depths.append(depths)
        else:
            depths = self.depths[-1]

        d = self.max_trials - trial - 1
        loc = d > depths
        d[loc] = depths[loc]
        
        logits = self.logits[-1]
        
        bern = Bernoulli(logits=logits[range(self.runs), d])

        res = bern.sample()
        valid = d > -1
        res[~valid] = nan

        return res




class BackInduction_full_planning(object):
    def __init__(self,
                 planet_confs, # Matrix of zeros and ones
                 runs=1, # number of parallel runs (i.e. agents). For each run, one can specify a different set of model parameters.
                 mini_blocks=1,
                 trials=1,
                 na=2, # no of actions
                 ns=6, # no of states
                 costs=None,
                 utility=None, # Utility of planet types. Can be set to rewards of the planets
                 planning_depth=1,
                 depths=None,
                 variable_depth=False):
        
        self.runs = runs
        self.nmb = mini_blocks
        self.trials = trials
        self.np = 2  # number of free model parameters

        self.depth = planning_depth  # maximal planning depth
        if depths is None:
            self.depths = [torch.tensor([planning_depth - 1]).repeat(self.runs)]
        if variable_depth:
            self.make_depth_transitions(rho=.8)
        else:
            self.make_depth_transitions()
        self.na = na  # number of actions
        self.ns = ns  # number of states

        # matrix containing planet type in each state
        self.pc = planet_confs
        
        if costs is not None:
            self.costs = costs
        else:
            self.costs = torch.tensor([-.2, -.5])

        if utility is not None:
            self.utility = utility
        else:
            self.utility = torch.arange(-2., 3., 1.)

        self.transitions = torch.tensor([4, 3, 4, 5, 1, 1])

    def set_parameters(self, trans_par=None, true_params=False):
        # [INPUT]
        # 1) None OR
        # 2) torch.tensor([a, b]), where a = ln(beta), b = theta OR
        # 3) torch.tensor([a, b], true_params=True), where a = beta, b = theta

        if trans_par is not None:
            if true_params:
                self.beta = (trans_par[..., 0]) # Response noise
                self.theta = trans_par[..., 1] # Bias term in sigmoid function fot action-selection
                   
            else:
                assert trans_par.shape[-1] == self.np
                self.beta = (trans_par[..., 0]).exp() # Response noise
                self.theta = trans_par[..., 1] # Bias term in sigmoid function fot action-selection

        else:
            self.beta = torch.tensor([10.]).repeat(self.runs)
            self.theta = zeros(self.runs)
          
        self.batch_shape = self.beta.shape

        self.tp_mean0 = torch.tensor([.9, .5]).expand(self.batch_shape + (2,)) 
        
        self.tau = torch.tensor(1e10).expand(self.batch_shape)

        # (Estimated) probability of successful jump
        self.tp_mean = [self.tp_mean0]

        # state transition matrices
        self.tm = []

        # expected state value
        self.Vs = []
        
        # expected action values
        self.Q = []

        # action value difference: Q(a=jump) - Q(a=move)
        self.D = []

        # response probability
        self.logits = []

    def make_depth_transitions(self, rho=1.):

        tm = torch.eye(self.depth).repeat(self.runs, 1, 1)
        if self.depth > 1:
            tm = rho*tm + (1-rho)*(ones(self.depth, self.depth) - tm)/(self.depth-1)

        self.tm_depths = tm

    def make_transition_matrix(self, p):
        # INPUT: p = probability of successful jump
        # Give p as tensor with #runs entries.
        na = self.na  # number of actions
        ns = self.ns  # number of states
        shape = self.batch_shape  # number of runs

        tm = zeros(shape + (na, ns, ns))

        # move left action - no tranistion uncertainty
        tm[..., 0, :-1, 1:] = torch.eye(ns-1)
        tm[..., 0, -1, 0] = 1

        # jump action - with varying levels of transition uncertainty
        tm[..., 1, -2:, 0:3] = (1 - p.reshape(shape+(1, 1)).expand(shape + (2, 3)))/2
        tm[..., 1, -2:, 1] = p.reshape(shape + (1,)).expand(shape + (2,))

        z = (1 - p.reshape(shape + (1,))).expand(shape + (3,))/2
        tm[..., 1, 2, 3:6] = z
        tm[..., 1, 0, 3:6] = z
        tm[..., 1, 1, 2:5] = z

        tm[..., 1, 2, 4] = p
        tm[..., 1, 0, 4] = p
        tm[..., 1, 3, 0] = (1 - p)/2
        tm[..., 1, 3, -2] = (1 - p)/2
        tm[..., 1, 3, -1] = p
        tm[..., 1, 1, 3] = p

        self.tm.append(tm)

    def compute_state_values(self, block):

        tm = self.tm[-1]  # transition matrix
        depth = self.depth  # planning depth
        shape = self.batch_shape

        utility = self.utility

        Vs = [torch.sum(utility * self.pc[:, block], -1).expand(shape+(self.ns,))]

        # action value difference: Q(a=jump) - Q(a=move)
        D = []

        R = torch.einsum('...ijkl,...il->...ikj', tm, Vs[-1]) + self.costs

        Q = R
        
        Qs = []
        
        for d in range(1, depth+1):
            
            Qs.append(Q)
            
            # compute Q value differences for different actions
            dQ = Q[..., 1] - Q[..., 0]

            # compute response probability
            p = (dQ * self.tau[..., None]).sigmoid()

            # set state value (=expected reward if one would start from this planet depending on depth iteration)
            Vs.append(p * Q[..., 1] + (1-p) * Q[..., 0])

            D.append(dQ)

            if d < depth:
                Q = torch.einsum('...ijkl,...il->...ikj', tm, Vs[-1]) + R

        self.Vs.append(torch.stack(Vs))
        self.D.append(torch.stack(D, -1))
        
        self.Q.append(torch.stack(Qs))

    def update_beliefs(self, block, trial, states, conditions, responses=None):
        # TODO: hier wurde temporal difference learning der transition uncertainty mit alpha entfernt
        # conditions is a (2 x #runs x #miniblocks) tensor
        # conditions[0, :, :] is noise condition (0: lownoise, 1: highnoise)
        # conditions[1, :, :] is no of steps in miniblocks
        self.noise = conditions[0]
        self.max_trials = conditions[1]

        subs = torch.arange(self.runs)
        #alpha = self.alpha

        if trial == 0:
            # update_transition_probability
            self.make_transition_matrix(self.tp_mean[-1][..., subs, self.noise])

        else:
        
            probs_new = self.tp_mean[-1][..., subs, self.noise]

            tp_mean = self.tp_mean[-1].clone()

            self.tp_mean.append(tp_mean)

            self.make_transition_matrix(probs_new)

        # set beliefs about state (i.e. location of agent) to observed states
        self.states = states

    def plan_actions(self, block, trial):

        self.compute_state_values(block)

        D = self.D[-1][..., range(self.runs), self.states, :]

        beta = self.beta[..., None]
        theta = self.theta[..., None]
        self.logits.append(D * beta + theta)
        
    def sample_responses(self, block, trial):
        if trial == 0 and block > 0:
            probs = self.tm_depths[range(self.runs), self.depths[0]]
            depths = Categorical(probs=probs).sample()
            loc = depths > self.max_trials - 1
            depths[loc] = self.max_trials[loc] - 1
            self.depths.append(depths)
        else:
            depths = self.depths[-1]

        d = self.max_trials - trial - 1
        loc = d > depths
        d[loc] = depths[loc]
        
        logits = self.logits[-1]
        
    
        bern = Bernoulli(logits=logits[range(self.runs), d])

        res = bern.sample()
        valid = d > -1
        res[~valid] = nan

        return res
    
    



class BackInduction_LPP(object):
    def __init__(self,
                 planet_confs, # Matrix of zeros and ones
                 runs=1, # number of parallel runs (i.e. agents). For each run, one can specify a different set of model parameters.
                 mini_blocks=1,
                 trials=1,
                 na=2, # no of actions
                 ns=6, # no of states
                 costs=None,
                 utility=None, # Utility of planet types. Can be set to rewards of the planets
                 planning_depth=1,
                 depths=None,
                 variable_depth=False):
        
        self.runs = runs
        self.nmb = mini_blocks
        self.trials = trials
        self.np = 2  # number of free model parameters

        self.depth = planning_depth  # maximal planning depth
        if depths is None:
            self.depths = [torch.tensor([planning_depth - 1]).repeat(self.runs)]
        if variable_depth:
            self.make_depth_transitions(rho=.8)
        else:
            self.make_depth_transitions()
        self.na = na  # number of actions
        self.ns = ns  # number of states

        # matrix containing planet type in each state
        self.pc = planet_confs
        
        if costs is not None:
            self.costs = costs
        else:
            self.costs = torch.tensor([-.2, -.5])

        if utility is not None:
            self.utility = utility
        else:
            self.utility = torch.arange(-2., 3., 1.)

        self.transitions = torch.tensor([4, 3, 4, 5, 1, 1])

    def set_parameters(self, trans_par=None, true_params=False):
        # [INPUT]
        # 1) None OR
        # 2) torch.tensor([a, b, c]), where a = ln(beta), b = theta, and c = logit(alpha) OR
        # 3) torch.tensor([a, b, c], true_params=True), where a = beta, b = theta, and c = alpha

        if trans_par is not None:
            if true_params:
                self.beta = (trans_par[..., 0]) # Response noise
                self.theta = trans_par[..., 1] # Bias term in sigmoid function fot action-selection
                   
            else:
                assert trans_par.shape[-1] == self.np
                self.beta = (trans_par[..., 0]).exp() # Response noise
                self.theta = trans_par[..., 1] # Bias term in sigmoid function fot action-selection

        else:
            self.beta = torch.tensor([10.]).repeat(self.runs)
            self.theta = zeros(self.runs)
          
        self.batch_shape = self.beta.shape

        self.tp_mean0 = torch.tensor([1., 1.]).expand(self.batch_shape + (2,)) 
        
        self.tau = torch.tensor(1e10).expand(self.batch_shape)

        # (Estimated) probability of successful jump
        self.tp_mean = [self.tp_mean0]

        # state transition matrices
        self.tm = []

        # expected state value
        self.Vs = []
        
        # expected action values
        self.Q = []

        # action value difference: Q(a=jump) - Q(a=move)
        self.D = []

        # response probability
        self.logits = []

    def make_depth_transitions(self, rho=1.):

        tm = torch.eye(self.depth).repeat(self.runs, 1, 1)
        if self.depth > 1:
            tm = rho*tm + (1-rho)*(ones(self.depth, self.depth) - tm)/(self.depth-1)

        self.tm_depths = tm

    def make_transition_matrix(self, p):
        # INPUT: p = probability of successful jump (100% in this agent)
        # Give p as tensor with #runs entries.
        na = self.na  # number of actions
        ns = self.ns  # number of states
        shape = self.batch_shape  # number of runs

        tm = zeros(shape + (na, ns, ns))

        # move left action - no tranistion uncertainty
        tm[..., 0, :-1, 1:] = torch.eye(ns-1)
        tm[..., 0, -1, 0] = 1

        # jump action - with varying levels of transition uncertainty
        tm[..., 1, -2:, 0:3] = (1 - p.reshape(shape+(1, 1)).expand(shape + (2, 3)))/2
        tm[..., 1, -2:, 1] = p.reshape(shape + (1,)).expand(shape + (2,))

        z = (1 - p.reshape(shape + (1,))).expand(shape + (3,))/2
        tm[..., 1, 2, 3:6] = z
        tm[..., 1, 0, 3:6] = z
        tm[..., 1, 1, 2:5] = z

        tm[..., 1, 2, 4] = p
        tm[..., 1, 0, 4] = p
        tm[..., 1, 3, 0] = (1 - p)/2
        tm[..., 1, 3, -2] = (1 - p)/2
        tm[..., 1, 3, -1] = p
        tm[..., 1, 1, 3] = p

        self.tm.append(tm)

    def compute_state_values(self, block):

        tm = self.tm[-1]  # transition matrix
        depth = self.depth  # planning depth
        shape = self.batch_shape

        utility = self.utility

        Vs = [torch.sum(utility * self.pc[:, block], -1).expand(shape+(self.ns,))]

        # action value difference: Q(a=jump) - Q(a=move)
        D = []

        R = torch.einsum('...ijkl,...il->...ikj', tm, Vs[-1]) + self.costs

        Q = R
        
        Qs = []
        
        for d in range(1, depth+1):
            
            Qs.append(Q)
            
            # compute Q value differences for different actions
            dQ = Q[..., 1] - Q[..., 0]

            # compute response probability
            p = (dQ * self.tau[..., None]).sigmoid()

            # set state value (=expected reward if one would start from this planet depending on depth iteration)
            Vs.append(p * Q[..., 1] + (1-p) * Q[..., 0])

            D.append(dQ)

            if d < depth:
                Q = torch.einsum('...ijkl,...il->...ikj', tm, Vs[-1]) + R

        self.Vs.append(torch.stack(Vs))
        self.D.append(torch.stack(D, -1))
        
        self.Q.append(torch.stack(Qs))

    def update_beliefs(self, block, trial, states, conditions, responses=None):
        # conditions is a (2 x #runs x #miniblocks) tensor
        # conditions[0, :, :] is noise condition (0: lownoise, 1: highnoise)
        # conditions[1, :, :] is no of steps in miniblocks
        self.noise = conditions[0]
        self.max_trials = conditions[1]

        subs = torch.arange(self.runs)
        #alpha = self.alpha

        if trial == 0:
            # update_transition_probability
            self.make_transition_matrix(self.tp_mean[-1][..., subs, self.noise])

        else:
        
            probs_new = self.tp_mean[-1][..., subs, self.noise]

            tp_mean = self.tp_mean[-1].clone()

            self.tp_mean.append(tp_mean)

            self.make_transition_matrix(probs_new)

        # set beliefs about state (i.e. location of agent) to observed states
        self.states = states

    def plan_actions(self, block, trial):

        self.compute_state_values(block)

        D = self.D[-1][..., range(self.runs), self.states, :]

        beta = self.beta[..., None]
        theta = self.theta[..., None]
        self.logits.append(D * beta + theta)
        
    def sample_responses(self, block, trial):
        if trial == 0 and block > 0:
            probs = self.tm_depths[range(self.runs), self.depths[0]]
            depths = Categorical(probs=probs).sample()
            loc = depths > self.max_trials - 1
            depths[loc] = self.max_trials[loc] - 1
            self.depths.append(depths)
        else:
            depths = self.depths[-1]

        d = self.max_trials - trial - 1
        loc = d > depths
        d[loc] = depths[loc]
        
        logits = self.logits[-1]
        
    
        bern = Bernoulli(logits=logits[range(self.runs), d])

        res = bern.sample()
        valid = d > -1
        res[~valid] = nan

        return res




class BackInduction_probDiscounting(object):
    def __init__(self,
                 planet_confs, # Matrix of zeros and ones
                 runs=1, # number of parallel runs (i.e. agents). For each run, one can specify a different set of model parameters.
                 mini_blocks=1,
                 trials=1,
                 na=2, # no of actions
                 ns=6, # no of states
                 costs=None,
                 utility=None, # Utility of planet types. Can be set to rewards of the planets
                 planning_depth=1,
                 depths=None,
                 variable_depth=False):
        
        self.runs = runs
        self.nmb = mini_blocks
        self.trials = trials
        self.np = 3  # number of free model parameters

        self.depth = planning_depth  # maximal planning depth
        if depths is None:
            self.depths = [torch.tensor([planning_depth - 1]).repeat(self.runs)]
        if variable_depth:
            self.make_depth_transitions(rho=.8)
        else:
            self.make_depth_transitions()
        self.na = na  # number of actions
        self.ns = ns  # number of states

        # matrix containing planet type in each state
        self.pc = planet_confs
        
        if costs is not None:
            self.costs = costs
        else:
            self.costs = torch.tensor([-.2, -.5])

        if utility is not None:
            self.utility = utility
        else:
            self.utility = torch.arange(-2., 3., 1.)

        self.transitions = torch.tensor([4, 3, 4, 5, 1, 1])

    def set_parameters(self, trans_par=None, true_params=False):
        # [INPUT]
        # 1) None OR
        # 2) torch.tensor([a, b]), where a = ln(beta), b = theta, and OR
        # 3) torch.tensor([a, b], true_params=True), where a = beta and b = theta

        if trans_par is not None:
            if true_params:
                self.beta = (trans_par[..., 0]) # Response noise
                self.theta = trans_par[..., 1] # Bias term in sigmoid function fot action-selection
                #self.k = (trans_par[..., 2]) # Discounting rate in the hyperbolic discounting model
                self.k = (trans_par[..., 2]) # assume k in [0,30]               
            else:
                assert trans_par.shape[-1] == self.np
                self.beta = (trans_par[..., 0]).exp() # Response noise
                self.theta = trans_par[..., 1] # Bias term in sigmoid function fot action-selection
                #self.k = (trans_par[..., 2]).exp() # Discounting rate in the hyperbolic discounting model
                self.k = 30*(trans_par[..., 2]).sigmoid() # k from [-inf,+inf] to [0,30]
        else:
            self.beta = torch.tensor([10.]).repeat(self.runs)
            self.theta = zeros(self.runs)
            self.k = zeros(self.runs)             

        self.batch_shape = self.beta.shape

        self.tp_mean0 = torch.tensor([.9, .5]).expand(self.batch_shape + (2,)) # used for planning

        self.tau = torch.tensor(1e10).expand(self.batch_shape)

        # (Estimated) probability of successful jump
        self.tp_mean = [self.tp_mean0]

        # state transition matrices
        self.tm = []

        # expected state value
        self.Vs = []
        
        # expected action values
        self.Q = []

        # action value difference: Q(a=jump) - Q(a=move)
        self.D = []

        # response probability
        self.logits = []
        

    def make_depth_transitions(self, rho=1.):

        tm = torch.eye(self.depth).repeat(self.runs, 1, 1)
        if self.depth > 1:
            tm = rho*tm + (1-rho)*(ones(self.depth, self.depth) - tm)/(self.depth-1)

        self.tm_depths = tm

    def make_transition_matrix(self, p):
        # INPUT: p = probability of successful jump
        # Give p as tensor with #runs entries.
        na = self.na  # number of actions
        ns = self.ns  # number of states
        shape = self.batch_shape  # number of runs

        tm = zeros(shape + (na, ns, ns))

        # move left action - no tranistion uncertainty
        tm[..., 0, :-1, 1:] = torch.eye(ns-1)
        tm[..., 0, -1, 0] = 1

        # jump action - with varying levels of transition uncertainty
        tm[..., 1, -2:, 0:3] = (1 - p.reshape(shape+(1, 1)).expand(shape + (2, 3)))/2
        tm[..., 1, -2:, 1] = p.reshape(shape + (1,)).expand(shape + (2,))

        z = (1 - p.reshape(shape + (1,))).expand(shape + (3,))/2
        tm[..., 1, 2, 3:6] = z
        tm[..., 1, 0, 3:6] = z
        tm[..., 1, 1, 2:5] = z

        tm[..., 1, 2, 4] = p
        tm[..., 1, 0, 4] = p
        tm[..., 1, 3, 0] = (1 - p)/2
        tm[..., 1, 3, -2] = (1 - p)/2
        tm[..., 1, 3, -1] = p
        tm[..., 1, 1, 3] = p

        self.tm.append(tm)

    def compute_state_values(self, block):

        tm = self.tm[-1]  # transition matrix
        depth = self.depth  # planning depth
        shape = self.batch_shape
        prob = self.tp_mean[-1][:,self.noise[0]]#[:,self.noise[0]]
        odds_against_success = (1 - prob)/prob
        if len(odds_against_success.shape) == 2:
            gamma = 1.0 / (1 + torch.einsum('...i, ...ij ->...ij', odds_against_success[:,self.noise[0]], self.k))
        elif len(odds_against_success.shape) == 1:
            gamma = 1.0 / (1 + torch.einsum('...i, ...i ->...i', odds_against_success[:], self.k))            

        utility = self.utility

        Vs = [torch.sum(utility * self.pc[:, block], -1).expand(shape+(self.ns,))] # Note (LG): Vs[0,:,:,:] initialized to planet rewards

        # action value difference: Q(a=jump) - Q(a=move)
        D = []

        R = torch.einsum('...ijkl,...il->...ikj', tm, Vs[-1]) + self.costs

        Q = R
        
        Qs = []
        
        for d in range(1, depth+1):
            
            Qs.append(Q)
            
            # compute Q value differences for different actions
            dQ = Q[..., 1] - Q[..., 0]  # TODO: Korrektur hier ohne Discounting
            
            # compute response probability
            p = (dQ * self.tau[..., None]).sigmoid() 

            # set state value
            Vs.append(p * Q[..., 1] + (1-p) * Q[..., 0])    
            
            # TODO: Korrektur im folgenden Block
            # use discounted deltaQ just for final Action Selection where self.D is used
            Qjump = torch.einsum('...i, ...ij ->...ij', gamma, Q[...,1]) # discounted
            dQ_discounted = Qjump - Q[..., 0]
            D.append(dQ_discounted) 

            if d < depth:
                Q = torch.einsum('...ijkl,...il->...ikj', tm, Vs[-1]) + R

        self.Vs.append(torch.stack(Vs))
        self.D.append(torch.stack(D, -1))
        
        self.Q.append(torch.stack(Qs))

    def update_beliefs(self, block, trial, states, conditions, responses=None):
        # TODO: hier wurde temporal difference learning der transition uncertainty mit alpha entfernt
        # conditions is a (2 x #runs x #miniblocks) tensor
        # conditions[0, :, :] is noise condition (0: lownoise, 1: highnoise)
        # conditions[1, :, :] is no of steps in miniblocks
        self.noise = conditions[0]
        self.max_trials = conditions[1]

        subs = torch.arange(self.runs)
        #alpha = self.alpha

        if trial == 0:
            # update_transition_probability
            self.make_transition_matrix(self.tp_mean[-1][..., subs, self.noise])

        else:
        
            probs_new = self.tp_mean[-1][..., subs, self.noise]

            tp_mean = self.tp_mean[-1].clone()

            self.tp_mean.append(tp_mean)

            self.make_transition_matrix(probs_new)

        # set beliefs about state (i.e. location of agent) to observed states
        self.states = states

    def plan_actions(self, block, trial):

        self.compute_state_values(block)

        D = self.D[-1][..., range(self.runs), self.states, :]

        beta = self.beta[..., None]
        theta = self.theta[..., None]
        self.logits.append(D * beta + theta)
        
    def sample_responses(self, block, trial):
        if trial == 0 and block > 0:
            probs = self.tm_depths[range(self.runs), self.depths[0]]
            depths = Categorical(probs=probs).sample()
            loc = depths > self.max_trials - 1
            depths[loc] = self.max_trials[loc] - 1
            self.depths.append(depths)
        else:
            depths = self.depths[-1]

        d = self.max_trials - trial - 1
        loc = d > depths
        d[loc] = depths[loc]
        
        logits = self.logits[-1]

        bern = Bernoulli(logits=logits[range(self.runs), d])

        res = bern.sample()
        valid = d > -1
        res[~valid] = nan

        return res


    

class BackInduction_LPP_and_probDiscounting(object):
    # lpp+discouting model from sass et al. (2025) with corrected bug in state value computation 
    def __init__(self,
                 planet_confs, # Matrix of zeros and ones
                 runs=1, # number of parallel runs (i.e. agents). For each run, one can specify a different set of model parameters.
                 mini_blocks=1,
                 trials=1,
                 na=2, # no of actions
                 ns=6, # no of states
                 costs=None,
                 utility=None, # Utility of planet types. Can be set to rewards of the planets
                 planning_depth=1,
                 depths=None,
                 variable_depth=False):
        
        self.runs = runs
        self.nmb = mini_blocks
        self.trials = trials
        self.np = 3  # number of free model parameters

        self.depth = planning_depth  # maximal planning depth
        if depths is None:
            self.depths = [torch.tensor([planning_depth - 1]).repeat(self.runs)]
        if variable_depth:
            self.make_depth_transitions(rho=.8)
        else:
            self.make_depth_transitions()
        self.na = na  # number of actions
        self.ns = ns  # number of states

        # matrix containing planet type in each state
        self.pc = planet_confs
        
        if costs is not None:
            self.costs = costs
        else:
            self.costs = torch.tensor([-.2, -.5])

        if utility is not None:
            self.utility = utility
        else:
            self.utility = torch.arange(-2., 3., 1.)

        self.transitions = torch.tensor([4, 3, 4, 5, 1, 1])

    def set_parameters(self, trans_par=None, true_params=False):
        # [INPUT]
        # 1) None OR
        # 2) torch.tensor([a, b, c]), where a = ln(beta), b = theta, and c = logit(alpha) OR
        # 3) torch.tensor([a, b, c], true_params=True), where a = beta, b = theta, and c = alpha

        if trans_par is not None:
            if true_params:
                self.beta = (trans_par[..., 0]) # Response noise
                self.theta = trans_par[..., 1] # Bias term in sigmoid function fot action-selection
                #self.alpha = (trans_par[..., 2]) # Learning rate for belief update     
                #self.k = (trans_par[..., 2]) # Discounting rate in the hyperbolic discounting model
                self.k = (trans_par[..., 2]) # assume k in [0,30]                           
            else:
                assert trans_par.shape[-1] == self.np
                self.beta = (trans_par[..., 0]).exp() # Response noise
                self.theta = trans_par[..., 1] # Bias term in sigmoid function fot action-selection
                #self.k = (trans_par[..., 2]).exp() # Discounting rate in the hyperbolic discounting model
                self.k = 30*(trans_par[..., 2]).sigmoid() # k from [-inf,+inf] to [0,30]                       
        else:
            self.beta = torch.tensor([10.]).repeat(self.runs)
            self.theta = zeros(self.runs)
            self.k = zeros(self.runs)             

        self.batch_shape = self.beta.shape

        self.tp_mean0 = torch.tensor([1., 1.]).expand(self.batch_shape + (2,))
        self.tp_mean1 = torch.tensor([.9, .5]).expand(self.batch_shape + (2,))        

        self.tau = torch.tensor(1e10).expand(self.batch_shape)

        # (Estimated) probability of successful jump
        self.tp_mean = [self.tp_mean0]

        # state transition matrices
        self.tm = []

        # expected state value
        self.Vs = []
        
        # expected action values
        self.Q = []

        # action value difference: Q(a=jump) - Q(a=move)
        self.D = []

        # response probability
        self.logits = []
        

    def make_depth_transitions(self, rho=1.):

        tm = torch.eye(self.depth).repeat(self.runs, 1, 1)
        if self.depth > 1:
            tm = rho*tm + (1-rho)*(ones(self.depth, self.depth) - tm)/(self.depth-1)

        self.tm_depths = tm

    def make_transition_matrix(self, p):
        # INPUT: p = probability of successful jump
        # Give p as tensor with #runs entries.
        na = self.na  # number of actions
        ns = self.ns  # number of states
        shape = self.batch_shape  # number of runs

        tm = zeros(shape + (na, ns, ns))

        # move left action - no tranistion uncertainty
        tm[..., 0, :-1, 1:] = torch.eye(ns-1)
        tm[..., 0, -1, 0] = 1

        # jump action - with varying levels of transition uncertainty
        tm[..., 1, -2:, 0:3] = (1 - p.reshape(shape+(1, 1)).expand(shape + (2, 3)))/2
        tm[..., 1, -2:, 1] = p.reshape(shape + (1,)).expand(shape + (2,))

        z = (1 - p.reshape(shape + (1,))).expand(shape + (3,))/2
        tm[..., 1, 2, 3:6] = z
        tm[..., 1, 0, 3:6] = z
        tm[..., 1, 1, 2:5] = z

        tm[..., 1, 2, 4] = p
        tm[..., 1, 0, 4] = p
        tm[..., 1, 3, 0] = (1 - p)/2
        tm[..., 1, 3, -2] = (1 - p)/2
        tm[..., 1, 3, -1] = p
        tm[..., 1, 1, 3] = p

        self.tm.append(tm)

    def compute_state_values(self, block):

        tm = self.tm[-1]  # transition matrix (here with LPP_1P_fix)
        depth = self.depth  # planning depth
        shape = self.batch_shape
        
        prob = self.tp_mean1[:,self.noise[0]]#[:,self.noise[0]]
        odds_against_success = (1 - prob)/prob
       
        if len(odds_against_success.shape) == 2:
            gamma = 1.0 / (1 + torch.einsum('...i, ...ij ->...ij', odds_against_success[:,self.noise[0]], self.k))
        elif len(odds_against_success.shape) == 1:
                    
            gamma = 1.0 / (1 + torch.einsum('...i, ...i ->...i', odds_against_success[:], self.k))            

        utility = self.utility

        Vs = [torch.sum(utility * self.pc[:, block], -1).expand(shape+(self.ns,))] 

        # action value difference: Q(a=jump) - Q(a=move)
        D = []

        R = torch.einsum('...ijkl,...il->...ikj', tm, Vs[-1]) + self.costs

        Q = R
        
        Qs = []
        
        for d in range(1, depth+1):
            
            Qs.append(Q)
            
            # compute Q value differences for different actions
            dQ = Q[..., 1] - Q[..., 0]  # TODO: Korrektur hier ohne Discounting
            
            # compute response probability
            p = (dQ * self.tau[..., None]).sigmoid() 

            # set state value (=expected reward if one would start from this planet depending on depth iteration)
            Vs.append(p * Q[..., 1] + (1-p) * Q[..., 0])    
            
            # TODO: Korrektur im folgenden Block
            # use discounted deltaQ just for final Action Selection where self.D is used -> for final planning step, discounted true transition probs are used for main transition (others also pruned)
            Qjump = torch.einsum('...i, ...ij ->...ij', gamma, Q[...,1]) # discounted
            dQ_discounted = Qjump - Q[..., 0]
            D.append(dQ_discounted) 

            if d < depth:
                Q = torch.einsum('...ijkl,...il->...ikj', tm, Vs[-1]) + R

        self.Vs.append(torch.stack(Vs))
        self.D.append(torch.stack(D, -1))
        
        self.Q.append(torch.stack(Qs))

    def update_beliefs(self, block, trial, states, conditions, responses=None):
        # conditions is a (2 x #runs x #miniblocks) tensor
        # conditions[0, :, :] is noise condition (0: lownoise, 1: highnoise)
        # conditions[1, :, :] is no of steps in miniblocks
        self.noise = conditions[0]
        self.max_trials = conditions[1]

        subs = torch.arange(self.runs)
        if trial == 0:
            # update_transition_probability
            self.make_transition_matrix(self.tp_mean[-1][..., subs, self.noise])

        else:
           
            probs_new = self.tp_mean[-1][..., subs, self.noise]

            tp_mean = self.tp_mean[-1].clone()

            self.tp_mean.append(tp_mean)

            self.make_transition_matrix(probs_new)

        # set beliefs about state (i.e. location of agent) to observed states
        self.states = states

    def plan_actions(self, block, trial):

        self.compute_state_values(block)

        D = self.D[-1][..., range(self.runs), self.states, :]

        beta = self.beta[..., None]
        theta = self.theta[..., None]
        self.logits.append(D * beta + theta)
        
    def sample_responses(self, block, trial):
        if trial == 0 and block > 0:
            probs = self.tm_depths[range(self.runs), self.depths[0]]
            depths = Categorical(probs=probs).sample()
            loc = depths > self.max_trials - 1
            depths[loc] = self.max_trials[loc] - 1
            self.depths.append(depths)
        else:
            depths = self.depths[-1]

        d = self.max_trials - trial - 1
        loc = d > depths
        d[loc] = depths[loc]
        
        logits = self.logits[-1]
        
  
        bern = Bernoulli(logits=logits[range(self.runs), d])

        res = bern.sample()
        valid = d > -1
        res[~valid] = nan

        return res