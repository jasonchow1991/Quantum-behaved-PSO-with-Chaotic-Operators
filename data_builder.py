# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:08:30 2020


RCPSP with sub-structure: definition of scenarios 


@author: zhtao
"""

import numpy as np
import os
import sys
import random
import math

class Activity():
    def __init__(self, ids, duration, resource):
        '''
        ids: activity id
        preced: a list of immediate precedence
        resource: a list of required resources 
        method: the method id it belongs to
        mandatory: if it is mandatory or not
        '''
        self.ids = ids
        self.duration = duration
        self.preced = []
        self.successor = []
        self.resource = resource
        self.method = 0
        self.mandatory = True



class Task():
    def __init__(self, ids, method):
        '''
        method: a list of alternative methods for a task
        '''
        self.ids = ids
        self.method = method


class Method():
    def __init__(self, ids, activity, task):
        '''
        activity: a list of activities included in a method
        task: task id that the method belongs to
        '''
        self.ids = ids
        self.activity = activity
        self.task = task

class Resource():
    def __init__(self, ids, max_demand):
        
        self.ids = ids
        self.max_demand = max_demand



class Data_generator():
    
    def __init__(self, resource_num, demand_limit, filename):
        
        self.resource_num = resource_num
        self.demand_limit = demand_limit
        self.filename = filename
        
    def mandatory_act(self, types):
        
        target_folder = self.filename + types
        
        scenario = random.choice(os.listdir(target_folder))
#        print(scenario)
        if types == 'J30':
            line_num = 32
        elif types == 'J60':
            line_num = 62
        elif types == 'J90':
            line_num = 92
        else:
            line_num = 122
            
        f = open(target_folder + '/' + scenario, 'r')
        
        list_1 = []
        list_2 = []
        resource = []
        for i, line in enumerate(f.readlines()):
            li = []
            if i >= 18 and i<= 18+line_num-1:
                data = line.strip().split(' ')
                for value in data:
                    if value == '':
                        pass
                    else:
                        li.append(int(value))
                list_1.append(li)
            elif i >= 18+line_num+4 and i <= 18+2*line_num+3:
                data = line.strip().split(' ')
                for value in data:
                    if value == '':
                        pass
                    else:
                        li.append(int(value))
                list_2.append(li)
            elif i == 18 + 2*line_num + 7:
                data = line.strip().split(' ')
                for value in data:
                    if value == '':
                        pass
                    else:
                        resource.append(int(value))
        
        Activity_list = []
        Resource_list = []
        
        # define the resources: if there are 4 types then use the original data, else we randomize 
        if self.resource_num == 4:
            for i, value in enumerate(resource):
                if value <= 10:
                    value = 10
                res = Resource(i, value)
                Resource_list.append(res)
        else:
            for i in range(self.resource_num):
                value = random.randint(int(0.8*self.demand_limit), self.demand_limit)
                res = Resource(i, value)
                Resource_list.append(res)
        # define the duration and resource demand of activity
        for i, value in enumerate(list_2):
            if self.resource_num == 4:
                activity = Activity(i, value[2], value[3:])
                Activity_list.append(activity)
            else:
                activity = Activity(i, value[2], [])
                for i in range(self.resource_num):
                    value = random.randint(0, Resource_list[i].max_demand)
                    activity.resource.append(value)
                Activity_list.append(activity)
        # define the immediate precedence for activity
        for i, value in enumerate(list_1):
            successors = value[3:]
            for act in successors:
                Activity_list[act-1].preced.append(i)
                Activity_list[i].successor.append(act-1)
                
        return Activity_list, Resource_list
    
    def alter_method(self, Activity_list, Resource_list, task_num, method_file):

        # defining tasks and their corresponding methods
        Tasks = []
        for i in range(task_num):
            Methods = []
            method_num = random.choice([2,3])
            for j in range(method_num):
                method_data = random.choice(os.listdir(method_file))
#                print(method_data)
                f = open(method_file+'/'+method_data, 'r')
                list_3 = []
                for i, line in enumerate(f.readlines()):
                    li = []
                    if i >= 4:
                        da = line.strip().split(' ')
                        for value in da:
                            if value == '':
                                pass
                            else:
                                li.append(int(value))
                        list_3.append(li)
                Methods.append(list_3)
            Tasks.append(Methods)
        
        # select edge and insert tasks & methods into the mandatory network
        not_available = list(set(Activity_list[0].successor + Activity_list[-1].preced))
        
        available_list = list(set([i for i in range(len(Activity_list))]).difference(not_available))
        
        acts = []
        while len(acts) < task_num:
            start = random.choice(available_list)
            if Activity_list[start].successor:
                acts.append(start)
        
        resources = [res.max_demand for res in Resource_list]
        
        Task_list = []
        Method_list = []
        for i, act, ta in zip([t for t in range(task_num)], acts, Tasks):

            end_point = random.choice(Activity_list[act].successor)
            task = Task(i, [])
            for j, met in enumerate(ta):
                method = Method(len(Method_list), [], i)
                acti_list =[]
                met_list =[]
                for k in met:
                    if len(resources) == 4:
                        acti = Activity(len(Activity_list), k[0], k[1:5])
                    else:
                        acti = Activity(len(Activity_list), k[0], [])
                        for h in range(len(resources)):
                            va = random.randint(0, resources[h])
                            acti.resource.append(va)
                    acti.mandatory=False
                    acti.method = method.ids
                    Activity_list.append(acti)
                    acti_list.append(acti)
                    method.activity.append(acti.ids)
                met_list.append(method)
                # dummy nodes of method need to connect with start node and end node
                acti_list[0].preced.append(act)
                acti_list[0].resource = [0]*len(resources)
                acti_list[-1].successor.append(end_point)
                acti_list[-1].resource = [0]*len(resources)
                Activity_list[act].successor.append(acti_list[0].ids)
                Activity_list[end_point].preced.append(acti_list[-1].ids)
                # the precedence relation within the activiites in method
                for s, ac in enumerate(acti_list):                    
                    succ = met[s][6:]
                    for su in succ:
                        acti_list[su-1].preced.append(ac.ids)
                        ac.successor.append(acti_list[su-1].ids)                
                task.method.append(method.ids)
                Method_list.extend(met_list)
            Task_list.append(task)
            
        return Activity_list, Method_list, Task_list
                    

#       
            
        

                
            


if __name__ == '__main__':
    
    sys.path.append('C:/DU-Scratch/Anaconda3/paper/RCPSP/Datasets')
    
    filename = 'C:/DU-Scratch/Anaconda3/paper/RCPSP/Datasets/'
    
    method_file = 'C:/DU-Scratch/Anaconda3/paper/RCPSP/Datasets/Method_data'
    
    random.seed(30)
    
    generator = Data_generator(4, 12, filename)
    
    Activity_list, Resource_list = generator.mandatory_act('J30')
    
    
    Activity_list, Method_list, Task_list = generator.alter_method(Activity_list, Resource_list, 3, method_file)
    
    MSL = np.array([1,0,1])
    
    ASL = np.zeros(len(Activity_list))
    select_ = []
    for i, activity in enumerate(Activity_list):
        if activity.mandatory:
            ASL[i] = 1
    for i, value in enumerate(MSL):
        if value == 1:
            act = Method_list[i].activity
            ASL[act] = 1
    for i, value in enumerate(ASL):
        if value == 1:
            select_.append(i)
    
    def start_time(index):
        if len(Activity_list[index].preced) == 0:
            start = 0
        else:
            start = 0
            for i in Activity_list[index].preced:
                if i in select_:
                   start = max(start, start_time(i)+Activity_list[i].duration)
                else:
                   start = -math.inf
        return start
    
    start_list=[]
    for i in range(len(Activity_list)):
        start = start_time(i)
        start_list.append(start)
    from copy import deepcopy
    Activity_2 = []
    for i, acti in enumerate(Activity_list):
        ac = deepcopy(acti)
        if acti.duration == 0:
            pass
        else:
            ac.duration = round(random.randint(math.floor(0.7*acti.duration), math.ceil(1.3*acti.duration)))
            if ac.duration < 1:
                ac.duration = 1
        Activity_2.append(ac)
        
        
        












