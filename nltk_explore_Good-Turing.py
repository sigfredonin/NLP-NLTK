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
        logN = log(n)
        TG_counts_dist_log_log_small += [ (count, log(logN), ) ]

TG_counts_dist_log_many = []
for count in TG_counts_dist:
    n = TG_counts_dist[count]
    if n > 10:
        logN = log(n)
        TG_counts_dist_log_many += [ (count, logN, ) ]

TG_counts_dist_log_log_many = []
for count in TG_counts_dist:
    n = TG_counts_dist[count]
    if n > 10:
        logN = log(n)
        TG_counts_dist_log_log_many += [ (count, log(logN), ) ]

scatter(*zip(*TG_counts_dist_log_many))
title("Emma -Log trigram counts distribution, N > 9")
show()
