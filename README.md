Code to reproduce the results of the paper "Online Markov Decision Processes with Terminal Law Constraints".

To run an experiment save all files to a folder, install the requirements in requirements.txt and execute

`python main.py --algo ‘MDPP-K’ --reward_type 'obstacles' --n_iterations 100 --online True`

Variables:
* `algo` can be ‘MDPP-K’ (Mirror Descent for Periodic Policies - Known initial distributions), ‘MDPP-U’ (Mirror Descent for Periodic Policies - Unknown initial distributions), or ‘episodic’, where the last corresponds to the Bonus MD-CURL algorithm referenced in the main paper
* `reward_type` can be ‘entropy_max’ or ‘obstacles’;
* `n_iterations` indicates the number of iterations;
* `online` is True when we consider the online case with unknown dynamics (default case), or False when we consider the case with known dynamics (for example, when computing the optimal periodic policy to compute the periodic regret)

The code runs the experiments for the chosen algorithm and experiment. It creates a ‘results’ folder containing a subfolder ‘frames’ with the state distribution images for each time step in the final iteration and the regret plot.

The `knownp` folder contains the objective function values for an approximately optimal periodic policy computed under the assumption that the MDP is fully known, used to compute the periodic regret.
