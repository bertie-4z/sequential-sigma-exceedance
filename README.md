# Sequential Sigma Exceedance

Code intakes a complete time series and for a given lookback period, computes the frequency at which the z-score of the time series exceeds a certain sigma, given that it has already exceeded a lower sigma benchmark. 

Output types include a matrix of conditional probabilities (proxied by computed frequencies), a group of histograms, and a generic curve plot (with incremental sigmas as the x-axis). 

