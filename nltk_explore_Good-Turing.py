from nltk_init_emma import *

# ... to figure out how c and Nc are related ...
from matplotlib.pyplot import *
from math import log
TG_counts_dist_log_small = [ (count, log(TG_counts_dist[count]),)
                             for count in TG_counts_dist
                             if count < 100 ]

TG_counts_dist_log_log_small = []
for count in TG_counts_dist:
    n = TG_counts_dist[count]
    if count < 10:
        logC = log(count)
        logN = log(n)
        TG_counts_dist_log_log_small += [ (logC, logN, ) ]

TG_counts_dist_log_many = []
for count in TG_counts_dist:
    n = TG_counts_dist[count]
    if n > 9:
        logN = log(n)
        TG_counts_dist_log_many += [ (count, logN, ) ]

TG_counts_dist_log_log_many = []
for count in TG_counts_dist:
    n = TG_counts_dist[count]
    if n > 9:
        logC = log(count)
        logN = log(n)
        TG_counts_dist_log_log_many += [ (logC, logN, ) ]

# Use linear regression to model the relation between log(c) and log(Nc)
# Use points for Nc > 9
from scipy.stats import linregress
log_c, log_nc = zip(*TG_counts_dist_log_log_many)
a, b, r_value, p_value, std_err = linregress(log_c, log_nc)
nc_estimated = [ b + a * float(x) for x in log_c ]
plot(log_c, log_nc, 'o', label="(log c, log Nc)")
plot(log_c, nc_estimated, 'r', label='Fitted line')
title("Emma - Log log trigram counts distribution, Nc > 9")
ylabel("Log Nc")
xlabel("Log c")
legend()
show()
