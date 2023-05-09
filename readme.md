![Photo](options_stock.jpg "Options Stock")

# Simulating American and European Options with Python. 
- Jerald Achaibar, Sukriti Raut, Joseph Yedinak

---

The `data` folder contains the input and output data from the simulations. 

- In the `setup` folder you can find the original dataset used, Namely the historical options data for $SPY, $QQQ, and $IWM; and historical risk free rates for the relevant time period. 
- The `output` folder contains the output data from the simulations, in the `output_analysis.ipynb` file (below.)

In the `setup` folder, you can find the preliminary research that we base all of our findings on. 

- The `LSMC_intro.ipynb` file shows what the Least Squares Monte Carlo method is, and how it can be applied to financial instruments for pricing models. 

- The `American_optioins.ipynb` steps through the LSMC Algorithm step by step, to achieve the final result. 

- The `black-scholes.ipynb` file dives into the derivation of the Brownian Motion, Monte Carlo Method for pricing European options and a comparison with the analytical methods of the Black Scholes model. 

The `model.ipynb` contains the code that was used to set up and test the program that would run the simulations. 

The `Models.py` contains our final model including the algorithm for LSMC, the Monte Carlo methods, and analytical Black Scholes that is used to run the simulations. 

The `output_analysis.ipynb` file is used to observe and convey the results from the simulations 