#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from PP_environment.environment import Env
from scipy.stats import truncnorm
import pandas as pd
import numpy as np
import random
import pickle
import uuid
import time
import math
import os
import copy
from operator import add, sub, mul
from termcolor import colored
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.animation as FuncAnimation
from IPython import display
from IPython.display import HTML
from celluloid import Camera # getting the camera

## starting main program
start = time.time()



gridHeightList = [int(sys.argv[1])]
gridWidthList = [int(sys.argv[1])]
noAgentList = [int(sys.argv[2])]
noObsList = [int(sys.argv[3])]
eList = [int(sys.argv[4])]
LoopVal = int(sys.argv[5]) # defines how many times the code will run
neighborWeightsList = [float(sys.argv[6])]
attackPercentage = [int(sys.argv[7])]
display = sys.argv[8]
sleep = float(sys.argv[9])
try:
    mode = sys.argv[10]
except:
    mode = 'random'
    
env_type=sys.argv[11].lower()

if mode.lower()=='random':
    playModeList = {"Agent":'random', "Target":'static', "Obstacle":'random', "Freeway":'random'}
else:
    playModeList = {"Agent":'random', "Target":'static', "Obstacle":'static', "Freeway":'static'}
flag = 0 # flag = 0, neighbor zone enabled and flag = 1, neighbor zone disabled

noTarget = 1 # there is only one target
noFreeway = 1 # there is only one freeway/resting area
AttackerList = [random.sample(range(1,noAgentList[0]), math.ceil(noAgentList[0]*attackPercentage[0]/100))] # calculating attackers' list
print("Attackers: ", AttackerList)


fig, axs = plt.subplots((noAgentList[0])-len(AttackerList[0])+1,1, sharex=True)
camera = Camera(fig)

 # reward and penalties
actionReward = 0
obsReward = -1.5
freewayReward = 0.5
emptycellReward = 0
hitwallReward = -0.5
goalReward = 10

min_v = obsReward
max_v = goalReward

# hyper-parameters
alpha = 0.1 # RL learning rate
varepsilon = 0.1 # privacy budget


### Laplace-based LDP mechanism
def Lap(randomlist, max_v, min_v, varepsilon, alpha):
    p_dataset = []
    b = (((max_v-min_v)*alpha)/varepsilon)
    for val in range(len(randomlist)):
        p_val = val + np.random.laplace(0, b)
        while ((p_val<min_v) or (p_val>max_v)):
            p_val = val + np.random.laplace(0, b)
        p_dataset.append(p_val)
    return p_dataset


def attack(advisorQ, min_v, max_v, varepsilon, alpha, adviseeQ):
    # initialize some lists
    newQAttackerDegree =[]
    newQAttacker = []
    newQAttackerList = []
    
    # initialize the degree of poisoning and poisoningFlag
    degreeVal = 0  
    poisoningFlag = True
    
    # compute the noise scale
    b = ((max_v-min_v)*alpha)/varepsilon
    
    while poisoningFlag==True:
        degreeVal += 1
        func = lambda k: ((2*b**2)/(k**2 - b**2)) - np.log(k**2) + np.log((k**2) - (b**2)) - degreeVal
        k_initial_guess = b+1
        p = fsolve(func, k_initial_guess)

        k1 = p[0]
        theta = 0
        miu_attacker = (((b**2)*(theta - (2*k1))) - (theta * (k1**2))) /((b**2) - (k1**2))

        for Q in advisorQ:
            malNoise = np.random.laplace(miu_attacker, b) # sampling malicious random Laplace noise according to attack impact and scale
            AdvQ = Q + malNoise # adding malicious noise
            
            # applying Bounded Laplace Mechanism (BLM)
            while ((AdvQ<min_v) or (AdvQ>max_v)):
                noise = np.random.laplace(miu_attacker, b)
                AdvQ = Q + noise
            newQAttackerDegree.append(AdvQ)
            
        # Ensuring malicious advice flows on the opposite direction of the advisee’s best possible action
        if ((newQAttackerDegree[adviseeQ.index(max(adviseeQ))]<max(adviseeQ)) or (degreeVal>12)): # we considered, poisoning threshold, \tau_gamma = 12
            poisoningFlag = False
        else:
            pass
        
        item = newQAttackerDegree.copy()
        newQAttackerDegree = []
    return degreeVal, item


 # initializing lists for calculating and saving convergence values
diffAvg1 = []
diffAvg2 = []
diffAvg3 = []
diffAvg4 = []
diffAvg5 = []
diffAvg4_0 = []


# Main Loop
for CriteriaVal in range(len(gridWidthList)):
    print("##################### Criteria Value: "+str(CriteriaVal)+" #######################\n")
    Attacker = AttackerList[CriteriaVal]
    Behb_tot = [100000 for i in range(noAgentList[CriteriaVal])] # advisee's budget for seeking advice during experience harvesting (EH)
    Besb_tot = [10000 for i in range(noAgentList[CriteriaVal])] # advisors' budget for seeking advice during experience giving (EG)
    fileName = str(uuid.uuid4())[:5] # initializing unique filename for storing learning outcomes
    stepsListFinal = []
    stepAgentListFinal = []
    rewards_all_episodesFinal = []
    qtableListFinal = []
    diffAvg5 = []
    for countVal in range(LoopVal):
        gridWidth = gridWidthList[CriteriaVal]#10
        gridHeight = gridHeightList[CriteriaVal]#10
        playMode = playModeList
        noAgent = noAgentList[CriteriaVal]
        noObs = noObsList[CriteriaVal]
        neighborWeights = neighborWeightsList[CriteriaVal]

        ## initialize varaibles
        qtableList = []
        aPosList = []
        stateList = []
        rewardList = []
        doneList = []
        actionList = []
        nextStateList = []
        rewards_all_episodes = []
        visitCount = []

        ## Check if no of elements greater than the state space or not
        if (noAgent+noTarget+noObs+noFreeway)>= (gridHeight * gridWidth):
            print("Total number of elements (agents, targets, obstacles) exceeds grid position")
        else:
            # building environment
            env = Env(gridHeight, gridWidth, playMode, noTarget, noAgent, noObs, noFreeway)
            print('-------Initial Environment---------\n')
            env.render()
            print("\n")

        ## for each agent, initializing a Q-table with random Q-values
        for a in range(noAgent):
            qtableList.append(np.random.rand(env.stateCount, env.actionCount).tolist())

        ## hyperparameters
        totalEpisode = eList[CriteriaVal]
        gamma = 0.8 # discount factor
        epsilon = 0.08 #0.08 #exploration-exploitation
        intEpsilon = epsilon
        decay = 0.1 # decay of exploration-exploitation over episode
        stepsList = []
        alpha = 0.1 #learning rate

        ## Function for environment display starts----------------------------------------------
        def dispEnv(stateList, aPosList, noAgent, gridWidth, gridHeight, env, disp, flag):
            if disp == True:
                print('State of the Players: ', stateList, '\n' )
                print('\n Players Info: ---->')
                for a in range(noAgent):
                    print('Position Of Player '+str(a)+': ', aPosList[a])
                print('\n')

            neighborDict = env.neighbors(noAgent, aPosList, gridWidth, gridHeight, flag)  
            neighborPosList = []
            for a in range(noAgent):
                neighborsPrint = []
                indNeighbor = []
                for player in neighborDict[a]:
                    neighborsPrint.append("P"+str(aPosList.index(player)))
                    indNeighbor.append(aPosList.index(player))
                if disp == True:
                    print("Neighbor of P"+ str(a)+" :" + str(neighborsPrint))
                neighborPosList.append(indNeighbor)
                indNeighbor = []
            if disp == True:
                print('\n')
            return neighborPosList
        ## environment display function ends----------------------------------------------
        
        ## initialize visit count for each state
        for i in range(noAgent):
            visitCount.append([0 for x in range((gridWidth*gridHeight))])
            
        ## initialize current experience harvesting budget (EHB) and current experience sharing budget (ESB)
        Behb = Behb_tot.copy()
        Besb = Besb_tot.copy()
          
        stepAgentList = [[] for i in range(noAgent)]    
        degreeValFinal = []    
        ## training loop
        for i in range(totalEpisode):
            degreeValListEp =[[] for i in range(noAgent)]
            print("epoch #", i+1, "/", totalEpisode)
            tPosList, aPosList, stateList, rewardList, doneList, oPosList, fPosList, courierNumber = env.reset(playMode, noTarget, noAgent, noObs,
                                                                       noFreeway, gridWidth, gridHeight, i, CriteriaVal,countVal,neighborWeights,totalEpisode,LoopVal)
            rewards_current_episode =[0 for a in range(noAgent)]
            doneList = [[a,'False'] for a in range(noAgent)]
            
            # render environment at the begining of every episode
            print("--------------Episode: ", i, " started----------------\n")
            if display=='on':
                env.render()
                print("\n")
            
            steps = 0
            completedAgent = []
            stepAgent = [0 for i in range(noAgent)]
            
            # uncomment only one line from below three lines according to your preference
            while [0, 'True'] not in doneList: # ends when agent0 reaches goal
#             while any('False' in sl for sl in doneList): # ends when all agents reach goal
#             while not any('True' in sl for sl in doneList): # ends when any agent reaches goal

                actionList = []
                if steps>(gridWidth*100):
                    break # break out of the episode if number of steps is too large to reach the goal.
                else:
                    steps +=1
                    
                ## find out neighbors starts---------------------------------------------------
                neighborDict = env.neighbors(noAgent, aPosList, gridWidth, gridHeight, flag)  
                neighborPosList = []
                for a in range(noAgent):
                    neighborsPrint = []
                    indNeighbor = []
                    for player in neighborDict[a]:
                        if a != aPosList.index(player):
                            indNeighbor.append(aPosList.index(player))
                        uniqueIndNeighbor = [*set(indNeighbor)]
                    neighborPosList.append(uniqueIndNeighbor)
                    uniqueIndNeighbor = []

                ## find out neighbors ends---------------------------------------------------
                
                ## find which agents have completed
                completedAgent = [i for i, x in enumerate(doneList) if x[1]=='True']
                
                for a in range(noAgent):
                    if ((a in completedAgent) and (stepAgent[a]==0)):
                        stepAgent[a] = steps
                
                ## update visit count for this state and every agent
                for a in range(noAgent):
                    visitCount[a][stateList[a]] += 1
                
                # Experience harvesting (EH) and Experience Giving (EG) phase
                for a in range(noAgent):
                    ## calculate Pehc (experience harvesting confidence) based on visit count and budget. 
                    # If visit count is too high (i.e., >100000) or too low (<100) for any episode, set experience harvesting confidence to low (i.e., will not seek for advice)
                    if ((visitCount[a][stateList[a]]< 100) or (visitCount[a][stateList[a]]> 100000)):
                        Pehc = 0
                    else:
                        Pehc = (1/np.sqrt(visitCount[a][stateList[a]])) * (np.sqrt(Behb[a]/Behb_tot[a]))
                    
                    if ((Pehc > 0) and (Pehc < 0.1)) :
                        Behb[a] = Behb[a]-1
                        QNeighbor  = []
                        if a not in completedAgent:
                            neighborsOldQ = 0
                            neighborsOldQList = []
                            adviseeQ = qtableList[a][stateList[a]]
                            if neighborPosList[a] !=[]:  #if not empty list
                                for n in neighborPosList[a]:
                                    ## calculate Pesc (experience sharing confidence) based on visit count and budget
                                    if (visitCount[n][stateList[a]]> visitCount[a][stateList[a]]):
                                        Pesc = (1-(1/np.sqrt(visitCount[n][stateList[a]]))) * (np.sqrt(Besb[n]/Besb_tot[n]))
                                    else:
                                        Pesc = 0
                                    if Pesc > 0:
                                        Besb[n] = Besb[n]-1
                                        
                                        # incorporating LDP
                                        noisyQ = Lap(qtableList[n][stateList[a]], max_v, min_v, varepsilon, alpha)
                                        neighborsOldQ = noisyQ
                                        
                                        #### Attacking (if any attacker presents)
                                        if n in Attacker:
                                            if a not in Attacker:
                                                oldQAttacker = qtableList[n][stateList[a]].copy()
                                                degreeVal, neighborsOldQ = attack(oldQAttacker, min_v, max_v, varepsilon, alpha, adviseeQ)
                                                print("Advisee, P"+str(a)+" is receiving"+
                                                      colored(" Malicious", 'red', attrs=['reverse'])+
                                                      " advice from Advisor, P"+str(n)+" at step: "+str(steps))
                                                degreeValListEp[n].append(degreeVal)
                                            else:
                                                neighborsOldQ = neighborsOldQ
                                                print("Advisee, P"+str(a)+" is also an attacker as Advisor, P"+str(n))
                                        else:
                                            neighborsOldQ = neighborsOldQ
                                        
                                        neighborsOldQList.append(neighborsOldQ)
                                    else:
                                        neighborsOldQ = []
                                        neighborsOldQList.append(neighborsOldQ)
                                
                                # combining neighbors expereince
                                if any(neighborsOldQList):
                                    for i in range(4): # here 4 stands for four different actions
                                        elem = [item[i] for item in neighborsOldQList if item!=[]]
                                        
                                        # selecting the most appropiate advice
                                        QNeighbor.append(np.mean(elem))
                                            
                                    # Weighted expereince aggregation
                                    qtableList[a][stateList[a]] = [sum(x) for x in zip([i * neighborWeights for i in adviseeQ], 
                                                     [i * (1-neighborWeights) for i in QNeighbor])]
                                else:
                                    qtableList[a][stateList[a]] = adviseeQ
                  
                
                # 1. select best action
                if np.random.uniform() < epsilon:
                    for a in range(noAgent):
                        actionList.append(env.randomAction())
                else:
                    for a in range(noAgent):
                        actionList.append(qtableList[a][stateList[a]].index(max(qtableList[a][stateList[a]])))
                        
                soqList = []   
                for a in range(noAgent):
                    soq = copy.deepcopy(qtableList[a])
                    soqList.append(soq)
                
                # 2. take the action and observe next state & reward
                nextStateList, rewardList, doneList, oPosList, courierNumber = env.step(actionList, doneList, noTarget, noAgent, noObs, noFreeway,
                                                               actionReward, obsReward, freewayReward, emptycellReward,
                                                               hitwallReward, completedAgent, goalReward)

                # 3. Calculate self Q-value
                for a in range(noAgent):
                    if a not in completedAgent:
                        qtableList[a][stateList[a]][actionList[a]] = ((qtableList[a][stateList[a]][actionList[a]] * (1 - alpha)) + (alpha * (rewardList[a] + gamma * max(qtableList[a][nextStateList[a]]))))
                        rewards_current_episode[a] += rewardList[a]
                        stateList[a] = nextStateList[a]
                    else:
                        qtableList[a][stateList[a]][actionList[a]] = qtableList[a][stateList[a]][actionList[a]]
                        rewards_current_episode[a] += rewardList[a]
                        stateList[a] = nextStateList[a]

                snqList = []
                for a in range(noAgent):
                    snq = copy.deepcopy(qtableList[a])
                    snqList.append(snq)
                
                # calcuating \Delta Q of the first agent for convergence analysis
                for p in range(len(soq)):
                    for q in range(len(soq[p])):
                        diff = abs(soqList[0][p][q] - snqList[0][p][q])
                        diffAvg1.append(diff)
                    diffAvg2.append(sum(diffAvg1)/len(diffAvg1))
                    diffAvg1 = []
                diffAvg3.append(sum(diffAvg2)/len(diffAvg2))
                diffAvg2 = []
            
            degreeValFinal.append(degreeValListEp)
            degreeValListEp=[]
            diffAvg4.append(sum(diffAvg3)/len(diffAvg3))
            diffAvg3 = []
            
            epsilon -= decay*epsilon # decaying exploration-exploitation probability for future episodes
            
            stepsList.append(steps)
            rewards_all_episodes.append(rewards_current_episode)
            print("\nDone in", steps, "steps".format(steps))
            time.sleep(sleep)
            
            color = ['red','cyan','black', 'green', 'magenta', 'orange', 'yellow', 
                     'red','cyan','black', 'green', 'magenta', 'orange', 'yellow',
                    'red','cyan','black', 'green', 'magenta', 'orange', 'yellow']
            stepAgent[stepAgent.index(0)]= steps
            axsCount = 0
            for a in range(noAgent):
                stepAgentList[a].append(stepAgent[a])
                if a not in Attacker:
                    axs[axsCount].plot(stepAgentList[a], marker='.', color=color[axsCount])
                    axs[axsCount].set_xticks([i for i in range(len(stepAgentList[a]))][-1:])
                    axs[axsCount].set_ylabel('P'+str(a))
                    axs[axsCount].set_ylim(0,500)
                    axs[axsCount].grid()
                    axsCount = axsCount+1
            axs[0].set_title("Attackers are: "+"P"+str(Attacker))
            axs[(noAgent-len(Attacker))].plot(stepsList, marker='x', color='blue')
            axs[(noAgent-len(Attacker))].set_xticks([i for i in range(len(stepsList))][-1:])
            axs[(noAgent-len(Attacker))].set_ylabel('All')
            axs[(noAgent-len(Attacker))].grid()
            axs[(noAgent-len(Attacker))].set_ylim(0,500)
            
        stepsListFinal.append(stepsList)
        stepsList = []
        rewards_all_episodesFinal.append(rewards_all_episodes)
        rewards_all_episodes = []
        qtableListFinal.append(qtableList)
        qtableList = []
        diffAvg5.append(diffAvg4)
        diffAvg4 = []
        diffAvg4_0 =[]
    
    
    end = time.time()
    total_time = end-start
    print("Total Time taken: ",total_time) 
    

dvf = degreeValFinal
with open("./SG/"+str(env_type)+"/"+str(fileName)+"_PeLPA_DegreeVal", "wb") as Sd:   #Pickling
    pickle.dump(dvf, Sd)

sa = stepAgentList
with open("./SG/"+str(env_type)+"/"+str(fileName)+"_PeLPA_Step_AgentWise", "wb") as Spa:   #Pickling
    pickle.dump(sa, Spa)

s = stepsListFinal
with open("./SG/"+str(env_type)+"/"+str(fileName)+"_PeLPA_Step", "wb") as Sp:   #Pickling
    pickle.dump(s, Sp)

r = rewards_all_episodesFinal
with open("./Reward/"+str(env_type)+"/"+str(fileName)+"_PeLPA_Reward", "wb") as Rp:   #Pickling
    pickle.dump(r, Rp)

q = qtableListFinal
with open("./Qtable/"+str(env_type)+"/"+str(fileName)+"_PeLPA_Qtable", "wb") as Qp:   #Pickling
    pickle.dump(q, Qp)

c = diffAvg5
with open("./Convergence/"+str(env_type)+"/"+str(fileName)+"_PeLPA_convergence", "wb") as Cp:   #Pickling
    pickle.dump(c, Cp)


t = total_time
with open("./TG/"+str(env_type)+"/"+str(fileName)+"_PeLPA_Time", "wb") as Tp:   #Pickling
    pickle.dump(t, Tp)   


with open("./OutputFile/PeLPA.txt", "a") as myfile:
    myfile.write("FileName: "+str(fileName)+" : PeLPA, Time taken: "+str(total_time)+"\n | gridWidth: "+str(gridWidth)+" | gridHeight: "+str(gridHeight)+
                " | playMode: "+str(playMode)+" | noTarget: "+str(noTarget)+" | noAgent: "+str(noAgent)+
                " | noObs: "+str(noObs)+" | noFreeway: "+str(noFreeway)+
                " | neighborWeights: "+str(neighborWeights)+" | totalEpisode: "+str(totalEpisode)+" | gamma: "+str(gamma)+
                " | epsilon: "+str(intEpsilon)+" | decay: "+str(decay)+" | alpha: "+str(alpha)+
                " | obsReward: "+str(obsReward)+" | freewayReward: "+str(freewayReward)+" | emptycellReward: "+str(emptycellReward)+
                " | hitwallReward: "+str(hitwallReward)+" | Attacker: "+str(Attacker)+"\n\n\n")   


