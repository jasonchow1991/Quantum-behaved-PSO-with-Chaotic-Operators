# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:59:14 2020


The implementation of Quantum Particle Swarm Optimization Algorithm on RCPSP

reference_1: https://github.com/shiluqiang/QPSO_python/blob/master/QPSO_RBF_SVM.py
reference_2: https://github.com/tommyod/paretoset/blob/master/paretoset/algorithms_numpy.py


@author: zhtao
"""
import time
import numpy as np
import random
import math
from sklearn.preprocessing import MinMaxScaler
import sys
from copy import deepcopy
from data_builder import *
from tqdm import tqdm, trange
import argparse 
from NSGA import *


'''
1. Sigmoid, Softmax functions
2. Function of calculating the Sigma value 
3. Logistic (chaotic) Mapping for generating chaotic sequence
'''
def sigmoid(x):
    return 1/(1+np.exp(-x))


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex)
    return ex/sum_ex

def combinations(A, k):
    if k==0:
        return [[]]
    if len(A) == k:
        return[list(A)]
    result = [[A[0]] + c for c in combinations(A[1:], k-1)]
    result += combinations(A[1:],k)
    return result

def sigma(x):
    if len(x) == 2:
        if x[0]+x[1]==0:
            sigma = 1
        else:
          sigma = [(x[0]**2 - x[1]**2)/(x[0]**2 + x[1]**2)]          
    elif len(x) > 2:
        list_ = np.arange(len(x))
        sum_ = 0
        for i in list_:
            sum_ += (x[i])**2
        combination = combinations(list_, 2)
        sigma = np.zeros(len(combination))
        for i, comb in enumerate(combination):
            sigma[i] = (x[comb[0]]**2 - x[comb[1]]**2)/sum_
    else:
        msg = 'the size of x must greater or equal to 2'
        raise ValueError(msg)
    return sigma
        

def chaotic_logistic(r, x):
    return r*x*(1-x)


'''
The class of defining particle properties including the position & fitness updating mechanism for both traditional PSO and Quantum PSO
'''

class Particle():
    
    def __init__(self, guid, position, n_fit):
        
        self.guid = guid
        self.position = position
        self.n_class = len(position)
        self.n_col = len(position[0])
        self.velocity = np.random.rand(self.n_class, self.n_col)
        self.current_fitness = np.zeros(n_fit)
        self.pbest_fitness = np.zeros(n_fit)
        self.pbest = np.zeros((self.n_class, self.n_col))
        
    def normal_velocity_update(self, w, c1, c2, gbest):
        #gbest is global best fitness
        for i in range(self.n_class):
            for j in range(self.n_col):
                r1 = random.random()
                r2 = random.random()
                vel_cognitive = c1*r1*(self.pbest[i][j]-self.position[i][j])
                vel_social = c2*r2*(gbest[i][j] - self.position[i][j])
                self.velocity[i][j] = w*self.velocity[i][j] + vel_cognitive + vel_social
    
    def normal_position_update(self):
        
        for i in range(self.n_class):
            for j in range(self.n_col):
                self.position[i][j] = self.position[i][j] + self.velocity[i][j]
                
    def quantum_position_update(self, beta, mbest, gbest):
        
        gamma = random.uniform(0,1)
        mu = random.uniform(0,1)
        Pbest_ = np.zeros((self.n_class, self.n_col))
        for i in range(self.n_class):
            for j in range(self.n_col):
                Pbest_[i][j] = gamma*self.pbest[i][j] + (1-gamma)*gbest[i][j]
                if random.random() > 0.5:
                    self.position[i][j] = Pbest_[i][j] + beta*math.log(1/mu)*(abs(mbest[i][j]-self.position[i][j]))
                else:
                    self.position[i][j] = Pbest_[i][j] - beta*math.log(1/mu)*(abs(mbest[i][j]-self.position[i][j]))

     
    def fitness_update(self, Activity_list, Method_list, Task_list, Resource_list, re_cost):
        
        ST_list, ASL, MSL, PL = SGS(self.position[0], Activity_list, Method_list, Task_list, Resource_list)
        
        self.current_fitness = fitness_fun(ST_list, ASL, Activity_list, re_cost)
        
        k = 0
        for i in range(len(self.pbest_fitness)):
            if self.pbest_fitness[i] >= self.current_fitness[i]:
                k += 1
        # if k == n_fit, current pbest dominates new_fit, otherwise, update pbest = new_fit
        if k != len(self.pbest_fitness):
            self.pbest_fitness = self.current_fitness
            self.pbest = self.position
            
            
'''
The class of Quantum-PSO including:
1. Swarm Initialization
2. Parameter beta - changing with iterations
3. Swarm Evolution Mechanism including alternative local search operators
4. Elite Selection Methods
'''

class QPSO():
    
    def __init__(self, particle_num, particle_dim_a, particle_dim_b, beta_range, upper, lower, n_fit, max_length, iter_num):
        ''' particle_num: number of particles/swarm size
            particle_dim: dimension of particle position (particle_dim_a = 1, particle_dim_b = len(Activity_list) + len(Method_list))
            beta: hyper-parameter for position updating (0.6)
            iter_num: iteration numbers
            upper: upper bound
            lower: lower bound
            n_fit: number of fitness functions (single obj or multi-obj)
            max_length: is the maximum length of gbest archiver
        '''
        self.particle_num = particle_num
        self.particle_dim_a = particle_dim_a
        self.particle_dim_b = particle_dim_b
        self.beta_range = beta_range
        self.upper = upper
        self.lower = lower
        self.n_fit = n_fit
        self.max_length = max_length
        self.iter_num = iter_num
        self.scaler_1 = MinMaxScaler(feature_range=(lower, upper))
    
    def swarm_initial(self, r, chaotic=False):
        swarm = []        
        for i in range(self.particle_num):
            position = np.zeros((self.particle_dim_a, self.particle_dim_b))
            chao_num = 0.1
            for a in range(self.particle_dim_a):
                for b in range(self.particle_dim_b):
                    if chaotic:
                        chao_num = chaotic_logistic(r, chao_num)
                        position[a][b] = chao_num*(self.upper - self.lower) + self.lower
                    else:
                        rand_= random.random()
                        position[a][b] = rand_*(self.upper - self.lower) + self.lower
                                       
            particle = Particle(i, position, self.n_fit)
            swarm.append(particle)
        # archieve is the list of non-dominated global bests
        archive = [swarm[random.randint(0, self.particle_num-1)]]
        
        return swarm, archive
    
    def get_beta(self, current_iter):
        beta_min, beta_max = self.beta_range
        beta = beta_max - ((beta_max-beta_min)/self.iter_num)*current_iter
        return beta
    
    def position_regu(self, position):
        for a in range(self.particle_dim_a):
            for b in range(self.particle_dim_b):
                if position[a][b] < self.lower:
                    position[a][b] = self.lower
                elif position[a][b] > self.upper:
                    position[a][b] = self.upper
    
    def update(self, swarm, archive, select_type, Activity_list, Method_list, Task_list, Resource_list, re_cost, algorithm, **kwargs):
        mbest = np.zeros((self.particle_dim_a, self.particle_dim_b))
        for i in range(len(swarm)):
            mbest += swarm[i].pbest
        mbest = mbest/len(swarm)
        # gbest = self.leader_select(swarm, archive, select_type).position
        # particles positions update & clipping and pbest update
        new_pbest_list = []
        for i in range(len(swarm)):
            if algorithm == 'QPSO':
#                print('the algorithm is QPSO')
                gbest = self.leader_select(swarm, archive, select_type, particle=swarm[i]).position
                current_beta = self.get_beta(kwargs['current_iter'])
                swarm[i].quantum_position_update(current_beta, mbest, gbest)                
                self.position_regu(swarm[i].position)
                # swarm[i].position = self.scaler_1.fit_transform(swarm[i].position.transpose()).transpose()
                swarm[i].fitness_update(Activity_list, Method_list, Task_list, Resource_list, re_cost)
                # local search
                # self.Neighbour_search(swarm[i], swarm, archive, current_iter = kwargs['current_iter'], Activity_list = Activity_list, Method_list=Method_list, Task_list=Task_list, Resource_list=Resource_list, re_cost=re_cost)
                
                # Crossover and Mutation
                if len(archive) > 2:
                    p = random.random()
                    self.Chaotic_mutation(swarm[i], swarm, archive, r = kwargs['r'], Activity_list = Activity_list, Method_list=Method_list, Task_list=Task_list, Resource_list=Resource_list, re_cost=re_cost)
                    if p > 0.5:
                        self.Chaotic_mutation_2(swarm[i], r=kwargs['r'])
                        swarm[i].fitness_update(Activity_list, Method_list, Task_list, Resource_list, re_cost)
                    
                    # q = random.random()
                    # if p > 0.5:
                    #     self.Neighbour_search(swarm[i], swarm, archive, current_iter = kwargs['current_iter'], Activity_list = Activity_list, Method_list=Method_list, Task_list=Task_list, Resource_list=Resource_list, re_cost=re_cost) 
                    # if q > 0.7:
                    #     self.Chaotic_mutation(swarm[i], swarm, archive, r = kwargs['r'], Activity_list = Activity_list, Method_list=Method_list, Task_list=Task_list, Resource_list=Resource_list, re_cost=re_cost)                      
                    # else:
                    #     self.Poly_mutation(swarm[i])
                    #     swarm[i].fitness_update(Activity_list, Method_list, Task_list, Resource_list, re_cost)
            else:
#                print('the algorithm is PSO')
                swarm[i].normal_velocity_update(**kwargs, gbest=gbest)
                swarm[i].normal_position_update()
                self.position_regu(swarm[i].position)
                # swarm[i].position = self.scaler_1.fit_transform(swarm[i].position.transpose()).transpose()
                swarm[i].fitness_update(Activity_list, Method_list, Task_list, Resource_list, re_cost)
            
            new_pbest_list.append(swarm[i].pbest_fitness)
        # add the new nondominated solutions to archiver
        is_efficient = Non_dominated_sorting.pareto_efficient_set(np.array(new_pbest_list), return_mask=False)
        for i in is_efficient:
            archive.append(swarm[i])
        gbest_fit_list = []
        # print(len(archive))
        for i in range(len(archive)):
            gbest_fit_list.append(archive[i].pbest_fitness)
        # update the nondominance in archiver and delete solutions that are being dominated
        is_efficient_ = Non_dominated_sorting.pareto_efficient_set(np.array(gbest_fit_list), return_mask=False)
        # print(is_efficient_)
        archive_ = [archive[i] for i in is_efficient_]
        # print(len(archive_))
        gbest_fit_list = [gbest_fit_list[i] for i in is_efficient_]
        # calculating the crowding distance of gbests            
        crowding_distance = Non_dominated_sorting.crowding_distance(np.array(gbest_fit_list))
        # print(len(crowding_distance))
        sorted_crowding_inds = np.argsort(crowding_distance)
        if len(archive_) > self.max_length:
            # remove the solutions with smallest values (more crowded region), [::-1] reverses the list
            sorted_crowding_inds = sorted_crowding_inds[::-1][:self.max_length]
            archive = [archive_[i] for i in sorted_crowding_inds]
        else:
            archive = archive_
    
    def Neighbour_search(self, particle, swarm, archive, current_iter, **kwargs):
        ''' Neighbourhood Search'''
        # randomly select an individual from archive and one from swarm; 
        individual_1 = archive[random.randint(0, len(archive)-1)]
        # generate a new individual in neighbourhood with individual 1
        position = np.zeros((self.particle_dim_a, self.particle_dim_b))
        for a in range(self.particle_dim_a):
            for b in range(self.particle_dim_b):
                position[a][b] = individual_1.position[a][b] + 0.25*random.uniform(-1,1)*math.exp((-4*current_iter)/self.iter_num)*(self.upper-self.lower)
                if position[a][b] < self.lower:
                    position[a][b] = self.lower 
                elif position[a][b] > self.upper:
                    position[a][b] = self.upper
    
        # position = self.scaler_1.fit_transform(position.transpose()).transpose()
        individual_2 = Particle(swarm[-1].guid+1, position, self.n_fit)
        individual_2.fitness_update(**kwargs)
        # if individual_3 dominates individual_2, then replace
        if Non_dominated_sorting.is_dominance(individual_2.pbest_fitness, particle.pbest_fitness):
            particle.position = individual_2.position
            particle.pbest = individual_2.pbest
            particle.pbest_fitness = individual_2.pbest_fitness
            particle.current_fitness = individual_2.current_fitness
        else:
            if random.random() > 0.5:
                particle.position = individual_2.position
                particle.pbest = individual_2.pbest
                particle.pbest_fitness = individual_2.pbest_fitness
                particle.current_fitness = individual_2.current_fitness
    
    ''' Chaotic Crossover Operator'''
    def Chaotic_mutation(self, particle, swarm, archive, r, **kwargs):
        #generate chaotic variables
        # x = random.random()
        li = [0.25, 0.5, 0.75]
        eta = 0.1
        while True:
            x = random.random()
            if round(chaotic_logistic(r, x),2) not in li:
                eta = chaotic_logistic(r, x)
                break
        rho = 4*eta*(1-eta)
        # randomly select two individuals from swarm and two from archive
        num_1, num_2 = random.sample([i for i in range(len(swarm))], 2)
        num_3, num_4 = random.sample([i for i in range(len(archive))], 2)
        # generate a new individual
        position = np.zeros((self.particle_dim_a, self.particle_dim_b))
        for a in range(self.particle_dim_a):
            for b in range(self.particle_dim_b):
                position[a][b] = particle.position[a][b] + eta*(archive[num_3].position[a][b] - swarm[num_1].position[a][b]) + rho*(archive[num_4].position[a][b]-swarm[num_2].position[a][b])
                if position[a][b] < self.lower:
                    position[a][b] = self.lower
                elif position[a][b] > self.upper:
                    position[a][b] = self.upper
        
        # position = self.scaler_1.fit_transform(position.transpose()).transpose()
        individual = Particle(len(swarm), position, self.n_fit)
        individual.fitness_update(**kwargs)
        if Non_dominated_sorting.is_dominance(individual.pbest_fitness, particle.pbest_fitness):
            particle.position = individual.position
            particle.pbest = individual.pbest
            particle.pbest_fitness = individual.pbest_fitness
            particle.current_fitness = individual.current_fitness
            
        else:
            # self.Poly_mutation(particle)
            # particle.fitness_update(**kwargs)
            if random.random() > 0.7:
                particle.position = individual.position
                particle.pbest = individual.pbest
                particle.pbest_fitness = individual.pbest_fitness
                particle.current_fitness = individual.current_fitness
    
    def Chaotic_mutation_2(self, particle, r):
        for a in range(self.particle_dim_a):
            for b in range(self.particle_dim_b):
                eta = (particle.position[a][b] - self.lower)/(self.upper-self.lower)
                rho = chaotic_logistic(r, eta)
                particle.position[a][b] = rho*(self.upper - self.lower) + self.lower
                if particle.position[a][b] < self.lower:
                    particle.position[a][b] = self.lower
                elif particle.position[a][b] > self.upper:
                    particle.position[a][b] = self.upper
   
    ''' Polynomial mutation operator '''
    def Poly_mutation(self, particle):
        for a in range(self.particle_dim_a):
            for b in range(self.particle_dim_b):
                u, delta = self.__get_delta()
                if u < 0.5:
                    particle.position[a][b] = delta*(particle.position[a][b] - self.lower)
                else:
                    particle.position[a][b] = delta*(self.upper - particle.position[a][b])
                if particle.position[a][b] < self.lower:
                    particle.position[a][b] = self.lower
                elif particle.position[a][b] > self.upper:
                    particle.position[a][b] = self.upper
                    
    def __get_delta(self, mutation_param=5):
        u = random.random()
        if u < 0.5:
            return u, (2*u)**(1/(mutation_param + 1)) - 1
        return u, 1 - (2*(1-u))**(1/(mutation_param + 1))
    
    ''' Elite Selection (Global best selection) based on Crowding distance and Sigma Value '''
    def leader_select(self, swarm, archive, types, particle=None):
        
        ids = 0
        
        if types == 'Crowding Distance':
            # select the one with higer CD value (less crowded region)
            gbest_fit_list = [archive[i].pbest_fitness for i in range(len(archive))]
            crowding_distance = Non_dominated_sorting.crowding_distance(np.array(gbest_fit_list))

            select_prob = softmax(crowding_distance)
            
            if len(select_prob) == 1:
                ids = 0
            else:
                if np.isnan(select_prob).any():
                    ids = np.random.choice(len(archive), 1)
                    ids = int(ids)
                else:
                    ids = np.random.choice(len(archive), 1, p=select_prob)
                    ids = int(ids)

                                
        if types == 'Sigma Method':
            # calculate the sigma value of each particle in archive and corresponding particels in swarm
            sigma_archive = [sigma(archive[i].pbest_fitness) for i in range(len(archive))]
            ind_list = [archive[i].guid for i in range(len(archive))]
            sigma_swarm = [sigma(swarm[i].current_fitness) for i in ind_list]
            if len(sigma_archive) == 1:
                ids = 0
            else:
                # calculate the distance
                dist = [math.sqrt(sum([(a-b)**2 for a, b in zip(sigma_archive[i], sigma_swarm[i])])) for i in range(len(archive))]
                # select the particle with minimum distance
                ids = np.argmin(dist)
                
        if types == 'Sigma Method 2':
            sigma_archive = [sigma(archive[i].pbest_fitness) for i in range(len(archive))]
            sigma_part = sigma(particle.current_fitness)
            if len(sigma_archive) == 1:
                ids = 0
            else:
                dist = [math.sqrt(sum([(a-b)**2 for a, b in zip(sigma_archive[i], sigma_part)])) for i in range(len(archive))]
                ids = np.argmin(dist)
                        
            
        return archive[ids]
            


         
'''Non-dominated Sorting class including finding out pareto optimal set and calculating crowd distance '''        

class Non_dominated_sorting():    
    
    @staticmethod
    def is_dominance(fitness_p, fitness_q): # determining if p dominates q
        
        
        dominance = False
        for i in range(len(fitness_p)):
            if fitness_p[i] < fitness_q[i]:
                return False
            if fitness_p[i] > fitness_q[i]:
                dominance = True
        return dominance 
    
    @staticmethod
    def pareto_efficient_set(data, return_mask=True):
        ''' data is an array with size (n_points, n_fit) 
            Return_mask: returning the True/false of pareto efficiency for each point
                        otherwise returning the indices of pareto efficient points.
        '''
        is_efficient = np.arange(data.shape[0])
        n_points = data.shape[0]
        next_point_index = 0
        while next_point_index < len(data):
            nondominate_mask = np.any(data<data[next_point_index], axis=1)
            nondominate_mask[next_point_index] = True
            is_efficient = is_efficient[nondominate_mask]
            data = data[nondominate_mask]
            next_point_index = np.sum(nondominate_mask[:next_point_index])+1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient
        
    @staticmethod
    def pareto_efficient_2(data):
        is_efficient = np.ones(data.shape[0], dtype=bool)
        for i, value in enumerate(data):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(data[is_efficient]<value, axis=1)
                is_efficient[i] = True
        return is_efficient
    
    @staticmethod
    def crowding_distance(data):
        msg = 'data argument must be np.ndarray of shape (n_points, n_fit)'
        if not isinstance(data, np.ndarray):
            raise ValueError(msg)
        if not data.ndim == 2:
            raise ValueError(msg)
            
        n_points, n_fit = data.shape
        if n_points <= 2:
            return np.ones(n_points)
        
        # prepare the distance matrix
        distances = np.zeros_like(data, dtype=np.float)
        
        arange_objectives = np.arange(n_fit)
    
        sorted_inds_reversed = np.zeros(data.shape, dtype=int)
        
        # sort every column individually
        sorted_inds = np.argsort(data, axis=0)
        sorted_matrix = np.take_along_axis(data, sorted_inds, axis=0)
        # compute the difference of a_i = a_{i+1} - a_{i-1}
        diffs = sorted_matrix[2:,:] - sorted_matrix[:-2, :]
        # compute max - min of each column
        max_min = sorted_matrix[-1,:] - sorted_matrix[0,:]
        # bottom and top rows are set to inf with only one neighbours
        max_row = (np.ones(n_fit)).reshape(1,-1)
        
        # create full matrix of difference
        diffs = np.vstack((max_row, diffs, max_row))
        assert diffs.shape == data.shape
        
        # prepare of matrix of reverse-sorted indices
        index = sorted_inds, arange_objectives
        
        sorted_inds_reversed[index] = np.tile(np.arange(n_points), (n_fit, 1)).T
        
        # distances (diffs) in orginal data order
        distances = diffs[sorted_inds_reversed, arange_objectives]
        
        # divide each column by max-min to normalize
        distances = np.divide(distances, max_min, out=distances)
        
        return distances.sum(axis=1)/n_fit
        

## schedule generation scheme (SGS) for proposed problem
           
def SGS(PSL, Activity_list, Method_list, Task_list, Resource_list):
    
    # PSL: priority selection list: [a, b, c, d, a_, b_, c_, d_]: [a,b,c,d] is the priority value; [a_,b_,c_,d_] is the selction
    #      of methods (0,1), the activity selection list can be obtained: mandatory activities + method activities
#    PSL = position[0].copy()
    PSL = PSL.copy()
    length = len(Activity_list)
    assert len(PSL) == length+len(Method_list)
    # priority value list
    RL = np.argsort(PSL[:length])
    for i, p in enumerate(RL):
        PSL[:length][p] = i + 1
    PL = np.array(PSL[:length], dtype=int)
    # repair the activity selection list & method selection list: ASL(activity selection list), MSL(method selection list)
    MSL = sigmoid(PSL[length:])

    # only one method is selected for each task
    for task in Task_list:
        methods = task.method.copy()
        method_id = task.method[np.argmax(MSL[methods])]
        MSL[method_id] = 1
        methods.remove(method_id)
        MSL[methods] = 0
    
    MSL = np.array(MSL, dtype=int)
    ASL = np.zeros(len(Activity_list))
    for i, activity in enumerate(Activity_list):
        if activity.mandatory:
            ASL[i] = 1
    for i, value in enumerate(MSL):
        if value == 1:
            act = Method_list[i].activity
            ASL[act] = 1
    ASL = np.array(ASL, dtype=int)
    select_act = []
    for i, value in enumerate(ASL):
        if value == 1:
            select_act.append(i)
#    print(select_act)
    # schedule generation process & precedence relation
    def start_time(index):
        if len(Activity_list[index].preced) == 0:
            start = 0
        else:
            start = 0
            for i in Activity_list[index].preced:
                if i in select_act:
                    start = max(start, start_time(i)+Activity_list[i].duration)
        return start
    
#    initial_list = [start_time(i) for i in range(len(Activity_list))]
#    print(initial_list)
    # activities in progress at time t
    def P_t(start_time_list, activity_squence, t):
        pt = []
        for i, start in enumerate(start_time_list):
            if t >= start and t< start + Activity_list[activity_squence[i]].duration:
                pt.append(activity_squence[i])
#            if i in select_act:
#                if t >= start and t < start + Activity_list[i].duration:
#                    pt.append(i)
        return pt
        
    start_time_list = [0]
    activity_squence = [0]
    select_num = np.count_nonzero(ASL)
#    k=0
    
    while len(activity_squence) < select_num:
        
        
        successors = [Activity_list[i].successor.copy() for i in activity_squence]
        candidates = []
        for succ in successors:
            for s in succ:
                if ASL[s] == 1 and s not in activity_squence:
                    candidates.append(s)
        candidates = list(set(candidates))
        candidate_act = []
        for act in candidates:
            preced_=[]
            for pre in Activity_list[act].preced:
                if ASL[pre] == 1:
                    preced_.append(pre)
            if all(elem in activity_squence for elem in preced_):
                candidate_act.append(act)
                       
#        candidate_act = Activity_list[activity_squence[k]].successor.copy()
        while candidate_act:
            ind_ = np.argmin([PL[i] for i in candidate_act])
            ind = candidate_act[ind_]
            if ind in activity_squence:
                candidate_act.remove(ind)
                continue            
            start_ = start_time(ind)
            for a in Activity_list[ind].preced:
                if ASL[a] == 1 and a in activity_squence:
                    st = start_time_list[activity_squence.index(a)]
                    start_ = max(start_, st + Activity_list[a].duration)
            ti = start_
            resource_const = 0
            while resource_const < len(Resource_list):
                pt = P_t(start_time_list, activity_squence, ti)
                pt_ = pt + [ind]
                for i, re in enumerate(Resource_list):
                    res_sum = sum([Activity_list[act].resource[i] for act in pt_])
                    if re.max_demand >= res_sum:
                        resource_const += 1
                    else:
                        complete_time = []
                        for p in pt:
                            complete = start_time_list[activity_squence.index(p)] + Activity_list[p].duration
                            complete_time.append(complete)
                        ti = min(complete_time)
                        continue
            E_start = ti
            activity_squence.append(ind)
            start_time_list.append(E_start)
            candidate_act.remove(ind)
            
    
    ST_list = [-math.inf]*len(Activity_list)
    for start, acti in zip(start_time_list, activity_squence):
        ST_list[acti] = start
    
    
    return ST_list, ASL, MSL, PL
        
    

            
                

            
## Fitness function ==================================================

def fitness_fun(ST_list, ASL, Activity_list, re_cost):
    # re_cost: resources costs
    duration_list = [act.duration for act in Activity_list]
    
    completion_list = [a+b for a,b in zip(ST_list, duration_list)]
    
    fitness_1 = max(completion_list)
    
    fitness_2 = 0
    for i, activity in enumerate(Activity_list):
        if ASL[i] == 1:
            fitness_2 += sum([a*b for a,b in zip(activity.resource, re_cost)])*activity.duration
    
    return [fitness_1, fitness_2]



# Problem Solver =================================================================

def RCPSP_solver(problem, algorithm, args):
    
    Activity_list, Method_list, Task_list, Resource_list = problem
    
    if algorithm == 'QPSO' or algorithm == 'PSO':
        
        solver = QPSO(args.particle_num, args.particle_dim_a, args.particle_dim_b, args.beta_range, 
                      args.upper, args.lower, args.n_fit, args.max_length, args.iter_num)
        
        swarm, archive = solver.swarm_initial(args.r, chaotic=args.chaotic)
        
        for i in trange(int(args.iter_num), desc='Iteration'):
            solver.update(swarm, archive, args.select_type, Activity_list, Method_list, Task_list, 
                          Resource_list, args.re_cost, algorithm, current_iter=i, r=args.r, w=args.w, c1=args.c1, c2=args.c2)
        
        best=[]
        for particle in archive:
            best.append(particle.pbest_fitness)
            
        is_efficient_ = Non_dominated_sorting.pareto_efficient_set(np.array(best), return_mask=False)
        
        archive_ = [archive[i] for i in is_efficient_]
                
        solutions = []
        for sol in archive_:
            ST_list, ASL, MSL, PL = SGS(sol.position[0], Activity_list, Method_list, Task_list, Resource_list)
            PSL = np.concatenate((PL,MSL))
            solutions.append([ST_list, ASL, PSL])
    
    elif algorithm == 'NSGA-II':
        
        feature_length = args.particle_dim_b
        
        incident = Problem_initial(args.n_fit, [(args.lower, args.upper)], feature_length, Activity_list, Method_list, 
                                                 Task_list, Resource_list, args.re_cost, same_range=True)
        
        Evo = Evolution(incident, num_of_generations = args.iter_num, num_of_individuals=args.particle_num)
        
        # the archive_ here is the pareto front of NSGA
        archive_ = Evo.evolve()
        
        solutions = []
        for individual in archive_:
            ST_list, ASL, MSL, PL = SGS(np.array(individual.features), Activity_list, Method_list, Task_list, Resource_list)
            PSL = np.concatenate((PL,MSL))
            solutions.append([ST_list, ASL, PSL])
        
        
        
           
    
    return solutions, archive_
            

# Evaluation Matrics ==============================================================
    
def evaluation(problem, solution, re_cost, expect_object, size):
    '''
    Three criteria: solution feasibility rate; objetive functions, robustness
    solution: is the particle in archive 
    expect_scenario: duration of each activity is the average of big sample scenarios for calculating the robustness
    size: the size of big sample: s_2
    '''
    Activity_list, Method_list, Task_list, Resource_list = problem
    ST_list, ASL, PSL = solution
    ST_list_, _, _, _ = SGS(PSL, Activity_list, Method_list, Task_list, Resource_list)
    makespan, cost = fitness_fun(ST_list_, ASL, Activity_list, re_cost)
    expect_makespan, expect_cost = expect_object
    # print('makespan is {}'.format(makespan))
    # print('expected makespan is {}'.format(expect_makespan))
    # feasibility: see if the solution meets the precedence & resource constraints (0 means not violate; 1 means violates)
    feasibility = 0
    for i, activity in enumerate(Activity_list):
        if ASL[i] == 1:
            if activity.successor:
                for j in activity.successor:
                    if ASL[j] == 1 and ST_list[i] + activity.duration > ST_list[j]:
                        feasibility = 1
                        break
    # for i, start in enumerate(ST_list):
    #     if start >= 0:
    #         complete = start + Activity_list[i].duration
    #         pt = []
    #         for j, t in enumerate(ST_list):
    #             if j != i:
    #                 if t >= start and t < complete:
    #                     pt.append(j)
    #         for r, resource in enumerate(Resource_list):
    #             if sum([Activity_list[a].resource[r] for a in pt]) > resource.max_demand:
    #                 feasibility = 1
    #                 break
    # robustness: max(f-f^/f^, 0)
    robustness_1 = (max((makespan-expect_makespan)/expect_makespan, 0))**2
    robustness_2 = (max((cost-expect_cost)/expect_cost, 0))**2
    
    return feasibility, robustness_1, robustness_2
    
    
    


    
## SAA-Main (Sample Average Approximation) =======================================================

def SAA_main(Activity_list, Method_list, Task_list, Resource_list, algorithm, args):
    '''
    Activity_list: here is the original activity-list with expected duration
    '''
    # sample_size: the number of samples, each sample with s_1 scenarios
    # theta is the level of deviation
    # set_1: set of samples of realisations
    # set_2: set of a large sample with realisations
    # s_1: number of scenarios for each sample in set_1
    # s_2: number of scenarios for big sample set_2
    # s_3: number of scenarios randomly selected from each sample in set_1
    set_1 = []
    set_2 = []
    for i in range(args.sample_size):
        sample = []
        for j in range(args.s_1):
            scenario = []
            for acti in Activity_list:
                act = deepcopy(acti)
                if acti.duration == 0:
                    pass
                else:
#                    act.duration = int(round(np.random.normal(acti.duration, args.theta_1, 1)[0]))
                    act.duration = round(random.randint(math.floor((1-args.theta)*acti.duration), math.ceil((1+args.theta)*acti.duration)))
                    if act.duration < 1:
                        act.duration = 1
                scenario.append(act)
            sample.append(scenario)
        set_1.append(sample)
        
    for i in range(args.s_2):
        scenario_ = []
        for acti in Activity_list:
            act_ = deepcopy(acti)
            if acti.duration == 0:
                pass
            else:
#                act.duration = int(round(np.random.normal(acti.duration, args.theta_1, 1)[0]))
                act_.duration = round(random.randint(math.floor((1-args.theta)*acti.duration), math.ceil((1+args.theta)*acti.duration)))
                if act_.duration < 1:
                    act_.duration = 1
            scenario_.append(act_)
        set_2.append(scenario_)
    # solve every sample problem
    candidate_solu = []
    for sample in set_1:
        random_senarios = random.sample(sample, args.s_3)
        new_scenario = []
        for i, acti in enumerate(Activity_list):
            activi = deepcopy(acti)
            # form an average scenario with duration equals to the max of randomly selected scenarios
            activi.duration = max([sena[i].duration for sena in random_senarios])
            new_scenario.append(activi)
        problem = [new_scenario, Method_list, Task_list, Resource_list]
        solu, archive = RCPSP_solver(problem, algorithm, args)
        candidate_solu.extend(solu)
    if algorithm == 'NSGA-II':
        if len(candidate_solu) > args.max_length:
            candidate_solu = candidate_solu[:args.max_length]
                
    end_1 = time.time()
    # evaluate the candidate solutions in a big sample set_2
    average_scenario = []
    for i, acti in enumerate(Activity_list):
        activity = deepcopy(acti)
        activity.duraiton = sum([sena[i].duration for sena in set_2])/args.s_2
        average_scenario.append(activity)   
        
    evaluation_metric = []
    for candidate in tqdm(candidate_solu):
        ST_list, ASL, PSL = candidate
        ST_list_, _, _, _ = SGS(PSL, average_scenario, Method_list, Task_list, Resource_list)
        expect_makespan, expect_cost = fitness_fun(ST_list_, ASL, average_scenario, args.re_cost)
        expect_object = [expect_makespan, expect_cost]
        feasibility = 0
        robustness_1 = 0
        robustness_2 = 0
        for scenario in tqdm(set_2):
            problem = [scenario, Method_list, Task_list, Resource_list]
            feasib, robust_1, robust_2 = evaluation(problem, candidate, args.re_cost, expect_object, args.s_2)
            feasibility += feasib
            robustness_1 += robust_1
            robustness_2 += robust_2
        feasibility_rate = feasibility/args.s_2
        robustness = math.sqrt(robustness_1/args.s_2) + args.deta * math.sqrt(robustness_2/args.s_2)
        evaluation_metric.append([expect_makespan, expect_cost, feasibility_rate, robustness])
                
    # obtain the non-dominated solutions
    is_pareto = Non_dominated_sorting().pareto_efficient_set(np.array(evaluation_metric), return_mask=False)
    
    Final_solution = [candidate_solu[i] for i in is_pareto]
    
    evaluate = [evaluation_metric[i] for i in is_pareto]
    
    return Final_solution, evaluate, end_1
            
            
                
            
            
            



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--filename', default=None, type=str, help='file of data for RCPSP')
    parser.add_argument('--method_file', default=None, type=str, help='file of data for optional methods')
    parser.add_argument('--sample_size', default=30, type=int, help='sample number of generated sample group for SAA')
    parser.add_argument('--resource_num', default=4, type=int, help='the number of types of resources')
    parser.add_argument('--re_cost', default=[], nargs='+', help='list of unit cost for each type of resource')
    parser.add_argument('--demand_limit', default=12, type=int, help='maximum demand for generating resource demand when resource number is not 4')
    parser.add_argument('--task_num', default=4, type=int, help='the number of tasks in the problem')
    parser.add_argument('--types', default=None, type=str, help='the type of problems, J30, J60, J90, J120')
    parser.add_argument('--s_1', default=100, type=int, help='the number of scenarios in a sample in SAA')
    parser.add_argument('--s_2', default=1000, type=int, help='the number of scenarios in a big sample in SAA')
    parser.add_argument('--s_3', default=90, type=int, help='random sampling size for approximating the durations')
    parser.add_argument('--theta', default=0.8, type=float, help='uncertainty ratio for duration - uniform distribution')
    parser.add_argument('--theta_1', default=4, type=float, help='uncertainty ratio for duration- variance of normal distribution')
    parser.add_argument('--deta', default=1, type=float, help='robustness coefficient')
    parser.add_argument('--particle_num', default=100, type=int, help='number of particles in a swarm')
    parser.add_argument('--particle_dim_a', default=1, type=int, help='first dimension of particle')
    parser.add_argument('--particle_dim_b', default=32, type=int, help='second dimension of particle')
    parser.add_argument('--beta_range', default=[0.4, 1], help='the range of beta for adaptive beta updating')
    parser.add_argument('--beta', default=0.5, type=float, help='updating coefficient for QPSO')
    parser.add_argument('--upper', default=10, type=int, help='upper bound for initializing position of particles')
    parser.add_argument('--lower', default=1, type=int, help='lower bound for initializing position')
    parser.add_argument('--n_fit', default=2, type=int, help='number of objectives')
    parser.add_argument('--max_length', default=1000, type=int, help='maximum length of archiver')
    parser.add_argument('--iter_num', default=100, type=int, help='number of iterations')
    parser.add_argument('--select_type', default=None, type=str, help='leader selection method')
    parser.add_argument('--w', default=0.5, type=float, help='inertia weight for PSO')
    parser.add_argument('--c1', default=2, type=float, help='cognitive factor/coefficient for PSO')
    parser.add_argument('--c2', default=2, type=float, help='social factor/coefficient for PSO')
    parser.add_argument('--r', default=4, type=float, help='system parameter for chaotic number generation')
    parser.add_argument('--chaotic', action='store_true', help='whether use chaotic number generator')
    
    args = parser.parse_args()
        
    sys.path.append('C:/DU-Scratch/Anaconda3/paper/RCPSP/Datasets')
       
    args.filename = 'C:/DU-Scratch/Anaconda3/paper/RCPSP/Datasets/'
    
    args.method_file = 'C:/DU-Scratch/Anaconda3/paper/RCPSP/Datasets/Method_data'
    
    args.types = 'J30'
    
    args.select_type = 'Sigma Method 2'
    
    algorithm = 'QPSO'
    
    args.chaotic = True
    
    seed = 30
    random.seed(seed)
    np.random.seed(seed)
    
    generator = Data_generator(args.resource_num, args.demand_limit, args.filename)
    
    Activity_list, Resource_list = generator.mandatory_act(args.types)
    
    Activity_list, Method_list, Task_list = generator.alter_method(Activity_list, Resource_list, args.task_num, args.method_file)
    
    for i in range(len(Resource_list)):
        value = random.randint(5, 15)
        args.re_cost.append(value)
    
    args.particle_dim_b = len(Activity_list) + len(Method_list)
    
#    candidate_solu, average = SAA_main(Activity_list, Method_list, Task_list, Resource_list, algorithm, args)
    
#    solver = QPSO(args.particle_num, args.particle_dim_a, args.particle_dim_b, args.beta, args.upper, args.lower, args.n_fit, args.max_length)
#    
#    swarm, archive = solver.swarm_initial()
#        
#    solver.update(swarm, archive, args.select_type, Activity_list, Method_list, Task_list, Resource_list, args.re_cost)
    
#    problem = [Activity_list, Method_list, Task_list, Resource_list]
#    
#    solutions, archive = RCPSP_solver(problem, algorithm, args)
    
    start = time.time()
        
    Final_solution, evaluate, end_1 = SAA_main(Activity_list, Method_list, Task_list, Resource_list, algorithm, args)
    
    end = time.time()
    
    print('The processing time of algorithm is {}'.format(end_1-start))
    print('The total processing time is {}'.format(end-start))

    with open('C:/DU-Scratch/Anaconda3/paper/RCPSP/Results.txt', 'a') as file:
        file.write('\n')
        file.write('J30-1 Improved Algorithm: {} Activity:{} Solution:{} Tasks:{} Methods:{} Resources:{}, seed:{} \n'.format(algorithm, len(Activity_list),
                    len(Final_solution), len(Task_list), len(Method_list), len(Resource_list), seed))
        file.write('CPU Computation time: {} \n'.format(end_1-start))
        file.write('Total Computation time: {} \n'.format(end-start))
        file.write('Resource costs: {} \n'.format(args.re_cost))
        file.write('Uncertainty Ratio: {}\n'.format(args.theta))
        file.write('confidence level: {} \n'.format(args.s_3/args.s_1))
        for i in range(len(Final_solution)):
            file.write('Solution: {} \n'.format(i+1))
            file.write('ST_list: {} \n'.format(Final_solution[i][0]))
            file.write('ASL: {} \n'.format(list(Final_solution[i][1])))
            file.write('PSL: {} \n'.format(list(Final_solution[i][2])))
            file.write('Evaluation Results: {} \n'.format(evaluate[i]))
    
    
    
    
    































































