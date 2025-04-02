# Sequential Sigma Exceedance

Code intakes a complete time series and for a given lookback period, computes the frequency at which the z-score of the time series exceeds a certain sigma, given that a lower sigma benchmark has already been exceeded. 

Output types include a matrix of conditional probabilities (proxied by computed frequencies), a group of histograms, and a generic curve plot (with incremental sigmas as the x-axis). 

## Awaiting fixes:

-Increment input parameter
-Number of thresholds parameter (this is currently arbitrarily set at 7)


