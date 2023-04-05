# PeLPA: Privacy-exploiting Local Poisoning Attacks in Cooperative Multiagent Reinforcement Learning

This is the codification used in the ICMLC 2023 paper proposing PeLPA framework as an novel adversarial method for poisoning cooperative multiagent reinforcement learning (CMARL) algorithms locally. You are free to use all or part of the codes here presented for any purpose, provided that the paper is properly cited and the original authors properly credited. All the files here shared come with no warranties.


This project was built on Python 3.8. All the experiments are executed in the modified Predator-Prey (PP) domain, we included the version we used in the **Main/PP_environment** folder (slightly different from the standard PP domain). For the graph generation code you will need to install Jupyter Notebook (http://jupyter.readthedocs.io/en/latest/install.html).

## Abstract
Cooperative Multiagent Reinforcement Learning (CMARL) has gained popularity in solving complex tasks through cooperation and collaboration of the participating agents in a complex environment. However, since CMARL may involve confidential client information, Differential Privacy (DP) has been proposed to secure it from adversarial inference. Nonetheless, the additional DP-noise creates a new threat for knowledge poisoning attacks in CMARL, which has not been extensively studied in the literature. To address this gap, we propose an adaptive knowledge poisoning technique that an attacker can use to exploit the additional DP-noise, evade state-of-the-art anomaly detection techniques, and prevent optimal convergence of the CMARL model. We evaluate our attack on state-of-the-art anomaly detection approaches in terms of detection accuracy and validation loss. 


## Files
The folder **Main** contains our implementation of all algorithms and experiments

The folder **Main/PP_environment** contains the modified Predator-Prey environment (also called a Pursuit domain) we used for experiments

Finally, the folder **ProcessedFiles** contains already processed files for graph printing and data visualization

## How to use <br />
First, install python 3.8 from https://www.python.org/downloads/release/python-380/<br />
Then open up your command terminal/prompt to run the following commands sequentially<br />
1. python RandomInit.py G N O E L Nw Ap D S M Et
2. python PeLPA.py G N O E L Nw Ap D S M Et


where, <br />
G: Grid Height and Width (N x N)<br />
N: number of agents<br />
O: number of obstacles<br />
E: Total Episode<br />
L: number of times the code will run as a loop<br />
Nw: Neighbor weights [0,1]<br />
Ap: Attack Percentage [0,100]<br />
D: Display environment [on, off]<br />
S: Sleep (sec)<br />
M: Play mode [random, static]<br />
Et: Environment type [small, medium, large] <br />

<br />
Example:<br />

python RandomInit.py 15 20 5 2000 10 0.90 20 on 2 random <br />
python PeLPA.py 15 20 5 2000 10 0.90 20 on 2 random large <br />

<br /><br />
         
However, it might take a long time until all the experiments are completed. 
It may be of interest to run more than one algorithm at the same time if you have enough computing power. 
Also, note that, for each framework, if the agents do not attain goal within (GridSize*100) steps in a particular episode, the episode and environment will be reset to the next. <br /><br />

The **file name** associated with any experiment is appended into a log file (PeLPA.txt) that resides inside "Main/OutputFile" directory.
The results (Steps to goal (SG), Rewards, Convergence) of any experiment are stored categorically by file name in "Main/SG", "Main/Reward", "Main/Convergence" respectively as a pickle file.
<br />
**Graph Generation and Reproduction**
1. Open processing.py file from "Main/" folder. Edit line 21-37 according to your experiments. Then run following command
	_python processing.py episode_num sg_gap reward_gap conv_gap env_type_
	where <br />
		episode_num = number of episode<br />
		sg_gap = plotting gap between SG values<br />
		reward_gap = plotting gap between Reward values<br />
    conv_gap = plotting gap between Convergence values<br />
    env_type = environment type [Option: small/medium/large]<br />
Example: _python processing.py 5000 500 500 20 small_ <br /><br />

Your processed output will be stored inside the "Main/ProcessedOutput" folder in .csv format. Example output files are: ProcessedSG.csv, ProcessedReward.csv, ProcessedConvergence.csv<br /><br />
2. Then one-by-one run "Main/graph_SG.py", "Main/graph_reward.py", "Main/graph_convergence.py" through below example steps.<br /><br />
	a. Open Main/graph_SG.py and edit line 48-55 as per your experiment and graph generation preferences<br />
	b. run _python graph_SG.py episode_num gap env_type_   (example: _python graph_SG.py 5000 500 small_)<br /><br />
	c. Open Main/graph_reward.py and edit line 49-56 as per your experiment and graph generation preferences<br />
	d. run _python graph_reward.py episode_num gap env_type_  (example: _python graphGenerator_Reward.py 5000 500 small_)<br /><br />
	e. Open Main/graph_convergence.py and edit line 51-58 as per your experiment and graph generation preferences<br />
	f. run _python graph_convergence.py episode_num gap env_type_   (example: _python graph_convergence.py 5000 20 small_)<br /><br />
	
	
Your output graphs will be stored in "Main/SG.pdf", "Main/Reward.pdf", "Main/Convergence.pdf" <br /><br />

3. For convenience, we include a "ProcessedFiles" folder that is already populated by the results of our experiments. <br />
	Processed outputs are already in the "ProcessedFiles/ProcssedOutput" folder.<br /><br />
	Simply, run the following commands from **./ProcessedFiles folder** to see the graphs we have included in our paper<br/>
	
	python graph_SG.py 5000 500 small
	python graph_reward.py 5000 500 small
	python graph_convergence.py 5000 20
	
	
## Contact
For questions about the codification or paper, <br />please send an email to mdtamjidh@nevada.unr.edu or aralab2018@gmail.com.
