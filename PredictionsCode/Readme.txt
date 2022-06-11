Run the file main.py to generate observations and infer predictions from the observations.

STEP 1: Generates the observational dat (runs Gillespie algorithm and extracts set of discrete observations to be used for inference)
STEP 2: Generates the reference (restarts Gillespie algorithm multiple times from last observation)
STEP 3: Infers the Posterior(s) and draws samples from the Posterior(s)
STEP 4: Applies Pushforward to MAP and samples
STEP 5: Finalizes predictions and plots observations, predictions and reference

The steps can be run individually (if output from previous step is available in 'myfolder' specified in main.py).

If you find any error, please email t.zerenner@gmail.com.
