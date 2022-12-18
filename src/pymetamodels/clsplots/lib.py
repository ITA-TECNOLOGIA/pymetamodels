
import numpy as np
import scipy.stats as stats

def confidence_interval_normal_std(mean, std, N, porcentage = 0.95):

    """
    Given a set of numbers in array
    
    The refeerence range or confidence interval is given 
    according a Student's t-distribution (Normal distribution)
    
    http://en.wikipedia.org/wiki/Reference_range
    
    Return Min interval, Max interval, p_value
    
    P_value indicate how normal is the distribution
    
    """
    
    ### Main statidistical values 
    if N<3:
        print "Not enough data"
        return 0, 0
    
    ### T student interval
    porcen = (1.+porcentage) / 2.
    
    interval = stats.t._ppf(porcen, N-1)
    #print N, porcen, interval
    
    ### Calculation
    N_1 = ( N + 1. ) / N
    N_1 = N_1 ** 0.5
    
    plus_minus = interval * N_1 * np.abs(std)
    
    lst = [mean - plus_minus, mean + plus_minus]
    
    ### Normal test
    p_value = None
    
    return np.min(lst), np.max(lst), p_value

def confidence_interval_normal(array, porcentage = 0.95):

    """
    Given a set of numbers in array
    
    The refeerence range or confidence interval is given 
    according a Student's t-distribution (Normal distribution)
    
    http://en.wikipedia.org/wiki/Reference_range
    
    Return Min interval, Max interval, p_value
    
    P_value indicate how normal is the distribution
    
    """
    
    ### Main statidistical values 
    N = np.sum((array * 0.) + 1.)
        
    if N<3:
        print "Not enough data"
        return 0, 0
        
    mean = np.mean(array)
    std = np.std(array)
    
    ### T student interval
    porcen = (1.+porcentage) / 2.
    
    interval = stats.t._ppf(porcen, N-1)
    #print N, porcen, interval
    
    ### Calculation
    N_1 = ( N + 1. ) / N
    N_1 = N_1 ** 0.5
    
    plus_minus = interval * N_1 * np.abs(std)
    
    lst = [mean - plus_minus, mean + plus_minus]
    
    ### Normal test
    p_value = stats.normaltest(array)[1]
    
    return np.min(lst), np.max(lst), p_value
    
def _test_confidence_interval():

    sample = ((np.random.random_sample((11,)))) * 50
    sample = stats.norm.rvs(size = 100)
    
    mm = np.mean(sample)
    guess = mm + 2 * np.std(sample)
    print "Mean sample %.3f -- std %.4f -- mean+2*std %.3f" % (mm, np.std(sample), guess)
    
    print "Interval %.3f to %.3f normality %.4f" % (confidence_interval_normal(sample, porcentage = 0.99))
    
#_test_confidence_interval()    


def confidence_interval_log(array, porcentage = 0.95):
    """
    Given a set of numbers in array
    
    The refeerence range or confidence interval is given 
    according a Student's t-distribution (Log Normal distribution)
    
    http://en.wikipedia.org/wiki/Reference_range
    
    Return Min interval, Max interval, p_value
    
    P_value indicate how log normal is the distribution
    
    """
    
    ### Main statidistical values 
    N = np.sum((array * 0.) + 1.)
        
    if N<3:
        print "Not enough data"
        return 0, 0
        
    mean = np.mean(array)
    std = np.std(array)
    
    mean_log = np.mean(np.log(array))
    deviation = np.sum(np.abs(mean_log - np.log(array)) ** 2.) / (N-1)
    std_log = deviation ** 0.5
    
    ### T student interval
    porcen = (1.+porcentage) / 2.
    
    interval = stats.t._ppf(porcen, N-1)
    
    ### Calculation  
    N_1 = ( N + 1. ) / N
    N_1 = N_1 ** 0.5    
    
    plus_minus = interval * N_1 * std_log
    
    lst = np.array([mean_log - plus_minus, mean_log + plus_minus])
    lst = np.exp(lst)
    
    ### Check
    p_value = 0
    
    return np.min(lst), np.max(lst), p_value

def _test_confidence_interval_log():
    
    sample = np.array([5.5, 5.2, 5.2, 5.8, 5.6, 4.6, 5.6, 5.9, 4.7, 5.0, 5.7, 5.2])
    #sample = stats.norm.rvs(size = 100)
    #sample = ((np.random.random_sample((11,)))) * 50
    
    print "Interval %.3f to %.3f normality %.4f" % (confidence_interval_normal(sample, porcentage = 0.99))
    print "Interval log %.3f to %.3f normality %.4f" % (confidence_interval_log(sample, porcentage = 0.95))
    
_test_confidence_interval_log()    