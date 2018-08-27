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
c_c_star_est = []
for count in TG_counts_dist:
    n_c = TG_counts_dist[count]
    n_c_plus_one = TG_counts_dist[count + 1]
    if n_c_plus_one == 0: # there are no trigrams with count c+1
        log_n_c_plus_one = b + a * log(count + 1)   # estimate Nc+1
        n_c_plus_one = exp(log_n_c_plus_one)
    c_star = (count + 1) * (n_c_plus_one / n_c)
    c_c_star += [ ( count, c_star, ) ]
    # Recompute using estimated Nc and Nc+1
    log_n_c_est = b + a * log(count)                # estimate Nc
    n_c_est = exp(log_n_c_est)
    log_n_c_plus_one_est = b + a * log(count + 1)   # estimate Nc+1
    n_c_plus_one_est = exp(log_n_c_plus_one_est)
    c_star_est = (count + 1) * (n_c_plus_one_est / n_c_est)
    c_c_star_est += [ ( count, c_star_est, ) ]
TG_counts_dist_GT = dict(c_c_star)
TG_counts_dist_GT_ext = dict(c_c_star_est)
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

# Plot c vs. c*
c_sorted = sorted( [ count for count in TG_counts_dist ] )
c_cs_sorted = [ (c, TG_counts_dist_GT[c],) for c in c_sorted ]
x, y = zip(*c_cs_sorted)
plot(x[:100],y[:100])
c_max = c_sorted[99]
plot(range(c_max),range(c_max),"r")
title("Emma - trigram counts vs. Good-Turing discounted counts, first 100")
ylabel("c*")
xlabel("c")
show()

# Plot c vs c*/c
c_cs_c_ratio_sorted = [ (c, TG_counts_dist_GT[c]/c,) for c in c_sorted ]
x, y = zip(*c_cs_c_ratio_sorted)
plot(x[:100],y[:100])
plot(x[:100],y[:100],'o')
title("Emma - trigram count Good-Turing discounted ratios, first 100")
ylabel("c*/c")
xlabel("c")
show()

# Plot c vs. c* using estimated Nc and Nc+1
c_cs_est_sorted = [ (c, TG_counts_dist_GT_ext[c],) for c in c_sorted ]
x, y = zip(*c_cs_est_sorted)
plot(x[:100],y[:100])
c_max = c_sorted[99]
plot(range(c_max),range(c_max),"r")
title("Emma - trigram counts vs. Good-Turing discounted counts, est Nc, first 100")
ylabel("c*")
xlabel("c")
show()

# Plot c vs c*/c using estimated Nc and Nc+1
c_cs_est_c_ratio_sorted = [ (c, TG_counts_dist_GT_ext[c]/c,) for c in c_sorted ]
x, y = zip(*c_cs_est_c_ratio_sorted)
plot(x[:100],y[:100])
plot(x[:100],y[:100],'o')
title("Emma - trigram count Good-Turing discounted estimated ratios, first 100")
ylabel("c*/c")
xlabel("c")
show()


# List Trigram counts for Emma
outFileName = "Emma_trigrams_GT_List.txt"
