import time
import numpy as np
from timeit import timeit
from random import seed, choice, randrange, uniform, sample

seed(1)

# Hyperparameter ranges.
node_min = 5
node_max = 55
node_step = 5
loss_funcs = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error']
dropout_min = 0.1
dropout_max = 0.5
dropout_step = 0.1
activations = ['tanh', 'sigmoid', 'relu', 'swish']

def truncate(f):
    f_str = str(f)
    f_str = f_str[:3]
    return float(f_str)

def generate_sample():
    nodes = randrange(node_min, node_max, node_step)
    dropout = truncate(uniform(dropout_min, dropout_max))
    loss = choice(loss_funcs)
    activation = choice(activations)
    #print(f"Nodes: {nodes}, Dropout: {dropout}, Loss: {loss}, Activation: {activation}")
    return (nodes, dropout, loss, activation)

def possible_repeats(sample_count):
    return [generate_sample() for _ in range(sample_count)]

def no_repeats(sample_count):
    start = time.time()
    result_set = set()
    while len(result_set) < sample_count:
        result_set.add(generate_sample())
        if time.time() - start >= 600:
            break
    return result_set

def generate_all_samples():
    results = []
    for node_count in range(node_min, node_max, node_step):
        for dropout in np.linspace(dropout_min, dropout_max, 9):
            for loss in loss_funcs:
                for lstm_activation in activations:
                    for dense_activation in activations:
                        results.append((node_count, dropout, loss, lstm_activation, dense_activation))
    return results

def select_random_hyperparameters(num_of_samples):
    return sample(generate_all_samples(), num_of_samples)

def all_samples_while():
    results = []
    pass

if __name__ == "__main__":
    try:
        sample_count = int(input("Enter number of samples to generate: "))
    except ValueError:
        print("Must be int.")
        raise SystemExit
    n = 1
    start_rep = time.time()
    reps = possible_repeats(sample_count)
    #rep_time = timeit("possible_repeats(sample_count)", 'from __main__ import possible_repeats, sample_count', number=n)
    end_rep = time.time()

    start_no_rep = time.time() 
    no_reps = no_repeats(sample_count)
    #no_rep_time = timeit("no_repeats(sample_count)", 'from __main__ import no_repeats, sample_count', number=n)
    end_no_rep = time.time() 

    start_all = time.time() 
    samples = all_samples_for()
    end_all = time.time() 
    print(f" For loop time: {end_rep - start_rep}")
    print(f" While loop time: {end_no_rep - start_no_rep}")
    print(f" All time: {end_all - start_all}")

    print(len(set(reps)))
    print(len(no_reps))
    print(len(samples))

    #all_time = timeit("all_samples_for()", 'from __main__ import all_samples_for', number=n)
