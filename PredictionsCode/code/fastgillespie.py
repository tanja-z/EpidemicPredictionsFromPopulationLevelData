# Author: Francesco Di Lauro
# Year: 2020
# email: F.Di-Lauro@sussex.ac.uk
#
# Modified by Tanja Zerenner (restart option added) t.zerenner@gmail.com

import networkx as nx
import numpy as np
from heapq import *
from datetime import datetime
import pprint as pp


#0:00:23.466188
class Node(): 
    def __init__(self,index,status, time):
        self.index = index
        self.status = status
        self.rec_time = time
class Event():
    def __init__(self,node,time,action, source=None):
        self.time = time

        self.node = node
        self.action = action
        self.source=source
    def __lt__(self, other):
        '''
            This is read by heappush to understand what the heap should be about
        '''
        return self.time < other.time        

class fast_Gillespie():
    '''
    This algorithm is inspired by  Joel Miller's algorithm for Fast Gillespie described in the book
    Mathematics of Epidemics on Networks by Kiss, Miller, Simon, 2017, Springer. Section A.1.2 page 384
    '''
    def __init__(self, A, tau=1.0, gamma=2.0, i0=10, tauf=4, discretestep=500, restart = 1): 
        if type(A)==nx.classes.graph.Graph:
            self.N = nx.number_of_nodes(A)
            self.A = A
        else:
            raise BaseException("Input networkx object only.")

        # Model Parameters (See Istvan paper).
        self.tau = tau
        self.gamma = gamma
        self.tauf = tauf
        
        # Time-keeping.
        self.cur_time = 0
        #output time vector
        self.time_grid =np.linspace(0,tauf,discretestep)
        
        #Node numbers. Number of infected people at various timsteps
        self.I = np.zeros(discretestep)
        if restart == 0:   
            self.I[0] = i0
        elif restart == 1:
            self.I[0] = sum(i0) #TODO sum over state
            
        self.current_index=1

        #number of SI links
        self.SI=np.zeros(self.N+1) 
        #time in each state
        self.tk = np.zeros(self.N+1)
        # (this is saved for the ak curves)
        
        if restart == 0:   
            #node state is [0] if not infected and [1] if infected
            X = np.array([0]*(self.N-i0) +[1]*i0)
            #display randomly the initial infected nodes
            np.random.shuffle(X) #todo this needs to be state?!
        elif restart == 1: 
            X = i0
            
        #nodes initialisation
        self.nodes = [Node(i,'susceptible', 0) for i in range(self.N)] 
        #keeps count of how many infected, useful for self.I and self.SI updates
        self.num_I = 0
        #Queue of Events, here each node has its own event
        self.queue=[]
        self.times=[]
        self.infected=[]
        self.cur_time=0
        for index in np.where(X==1)[0]:
            event = Event(self.nodes[index],0,'transmit', source=Node(-1,'infected',0))
            heappush(self.queue,event)
        #network states; 0 = susceptible; 1 = infected
        self.state = np.zeros((self.N,discretestep))
        self.state[:,0] = X   
            
            
    def run_sim(self):
        '''first round outside to determine SI'''
        num_SI=0        
        while self.queue:
            '''
                condition to stop
            '''
            event = heappop(self.queue)
            #dt is used only to update SI
            '''
            If node is susceptible and it has an event it must be an infection
            '''
            if event.action=='transmit':
                if event.node.status =='susceptible':
                    dt = event.time -self.cur_time
                    #set new time accordingly
                    '''
                    AFTER finding dt you can update SI
                    '''
                    self.SI[self.num_I] += num_SI*dt
                    self.tk[self.num_I] += dt
                    num_SI +=self.process_trans(event.node, event.time)                
                    '''
                    check if time grid needs to be updated
                    '''
                    if self.cur_time <self.tauf:
                        while self.time_grid[self.current_index] <= self.cur_time:                    
                            self.I[self.current_index] = self.num_I
                            for ii in range(self.N):
                                   currentnode = self.nodes[ii]
                                   if currentnode.status == 'infected':
                                       self.state[ii,self.current_index] = 1
                            self.current_index +=1  
                    
                self.find_next_trans(event.source, event.node, event.time)
            else:
                dt = event.time -self.cur_time
                self.SI[self.num_I] += num_SI*dt
                self.tk[self.num_I] += dt
                num_SI +=self.process_rec(event.node,event.time)
                if self.cur_time <self.tauf:
                    while self.time_grid[self.current_index] <= self.cur_time:                    
                        self.I[self.current_index] = self.num_I
                        for ii in range(self.N):
                            currentnode = self.nodes[ii]
                            if currentnode.status == 'infected':
                                self.state[ii,self.current_index] = 1
                        self.current_index +=1      


        #self.I[self.current_index:] = self.I[self.current_index-1]
        self.I[self.current_index:] = self.num_I
        
        for ii in range(self.N):
               currentnode = self.nodes[ii]
               if currentnode.status == 'infected':
                   self.state[ii,self.current_index] = 1      



    def process_trans(self,node,time):
        '''
        utility for transmission events:
        it checks also the neighbours.
        Returns number of SI as well
        '''
        #self.times.append(time)
        self.cur_time=time
        self.num_I +=1
        '''
        if len(self.infected) >0:
            self.infected.append(self.infected[-1]+1)
        else:
            self.infected.append(1)
        '''    
        node.status='infected'
        
        r1 = np.random.rand()
        rec_time = time -1.0/self.gamma *np.log(r1)
        node.rec_time = rec_time
        
        if rec_time < self.tauf:
            event = Event(node,rec_time,'recover', None)
            heappush(self.queue,event)
        num_SI=0    
        for index in self.A.neighbors(node.index):
            neighbor = self.nodes[index]
            if neighbor.status=='susceptible':
                num_SI+=1
            else:
                num_SI-=1
            self.find_next_trans(source = node, target = neighbor, time = time)
        return num_SI
    def find_next_trans(self,source,target,time):
        if target.rec_time < source.rec_time:
            r1 = np.random.rand()
            trans_time = max(time,target.rec_time) -1.0/self.tau *np.log(r1)
            if trans_time < source.rec_time and trans_time<self.tauf:
                event = Event(node=target, time=trans_time,  action='transmit', source=source)
                heappush(self.queue,event)
                
    def process_rec(self, node, time):
        node.status='susceptible'
        node.rec_time = 0
        num_SI=0
        self.num_I -=1
        for index in self.A.neighbors(node.index):
            neighbor = self.nodes[index]
            if neighbor.status=='susceptible':
                num_SI-=1
            else:
                num_SI+=1
        #self.times.append(time)
        self.cur_time=time
        #self.infected.append(self.infected[-1]-1)
        return num_SI
    
