from nltk_init_emma import *

# ... to figure out how c and Nc are related ...
from matplotlib.pyplot import *
from math import log, exp

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
nc_estimated = [ b + a*x for x in log_c ]

plot(log_c, log_nc, 'o', label="(log c, log Nc)")
plot(log_c, nc_estimated, 'r', label='Fitted line')
title("Emma - Log log trigram counts distribution, Nc > 9")
ylabel("Log Nc")
xlabel("Log c")
legend()
show()

# Re-compute N-gram counts for Emma
c_c_star = []
for count in TG_counts_dist:
    n_c = TG_counts_dist[count]
    n_c_plus_one = TG_counts_dist[count + 1]
    if n_c_plus_one == 0: # there are no trigrams with count c+1
        log_n_c_plus_one = b + a * (log(count + 1))   # estimate Nc+1
        n_c_plus_one = round(exp(log_n_c_plus_one))
    c_star = (count + 1) * (n_c_plus_one / n_c)
    c_c_star += [ ( count, c_star, ) ]
TG_counts_dist_GT = dict(c_c_star)
TG_dist_GT = { gram : TG_counts_dist_GT[TG_dist[gram]] for gram in TG_dist}

# Compute some validations ...
N_from_N_grams = len(trigrams)
N_from_N_grams_dist = sum([ TG_dist[gram] for gram in TG_dist])
N_from_N_c = sum([ count * TG_counts_dist[count] for count in TG_counts_dist ])
print("Nummber of trigrams:", N_from_N_grams)
print("Sum of trigram counts:", N_from_N_grams_dist)
print("Sum of count occurrences (c*Nc):", N_from_N_c)
N1_from_list_Nc = TG_counts_dist[1]
N1_from_list_Trigram_counts = sum([ TG_dist[count] for count in TG_dist
                                    if TG_dist[count] == 1])
print("# trigrams occurring once:", N1_from_list_Trigram_counts)
print("Nc for c=1:", N1_from_list_Nc)
