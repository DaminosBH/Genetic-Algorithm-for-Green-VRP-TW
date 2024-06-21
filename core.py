# -*- coding: utf-8 -*-

'''gavrptw/core.py'''

import os
import io
import random
import time
from csv import DictWriter
from deap import base, creator, tools
from . import BASE_DIR
from .utils import make_dirs_for_file, exist, load_instance, merge_rules


def ind2route(individual, instance):
    '''gavrptw.core.ind2route(individual, instance)'''
    route = []
    vehicle_capacity = instance['vehicle_capacity']
    depart_due_time = instance['depart']['due_time']
    # Initialize a sub-route
    sub_route = []
    vehicle_load = 0
    elapsed_time = 0
    last_customer_id = 0
    for customer_id in individual:
        # Update vehicle load
        demand = instance[f'customer_{customer_id}']['demand']
        updated_vehicle_load = vehicle_load + demand
        # Update elapsed time
        service_time = instance[f'customer_{customer_id}']['service_time']
        return_time = instance['distance_matrix'][customer_id][0]
        updated_elapsed_time = elapsed_time + \
            instance['distance_matrix'][last_customer_id][customer_id] + service_time + return_time
        # Validate vehicle load and elapsed time
        if (updated_vehicle_load <= vehicle_capacity) and (updated_elapsed_time <= depart_due_time):
            # Add to current sub-route
            sub_route.append(customer_id)
            vehicle_load = updated_vehicle_load
            elapsed_time = updated_elapsed_time - return_time
        else:
            # Save current sub-route
            route.append(sub_route)
            # Initialize a new sub-route and add to it
            sub_route = [customer_id]
            vehicle_load = demand
            elapsed_time = instance['distance_matrix'][0][customer_id] + service_time
        # Update last customer ID
        last_customer_id = customer_id
    if sub_route != []:
        # Save current sub-route before return if not empty
        route.append(sub_route)
        print('Route of current individual')
        print(route)
    return route


def print_route(route, merge=False):
    '''gavrptw.core.print_route(route, merge=False)'''
    route_str = '0'
    sub_route_count = 0
    for sub_route in route:
        sub_route_count += 1
        sub_route_str = '0'
        for customer_id in sub_route:
            sub_route_str = f'{sub_route_str} - {customer_id}'
            route_str = f'{route_str} - {customer_id}'
        sub_route_str = f'{sub_route_str} - 0'
        if not merge:
            print(f'  Vehicle {sub_route_count}\'s route: {sub_route_str}')
        route_str = f'{route_str} - 0'
    if merge:
        print(route_str)


def eval_vrptw(individual, instance,a,b,c,d,e,f,g,v, unit_cost=1.0, init_cost=0, wait_cost=0, delay_cost=1, carbon_cost=0.1):
    '''gavrptw.core.eval_vrptw(individual, instance, unit_cost=1.0, init_cost=0, wait_cost=0,
        delay_cost=0)'''
    k,l,g1,q,r,u=1.27,0.0614,1,-0.0011,-0.00235,-1.33
    sub_route_emission_cost=0
    total_distance=0
    total_cost = 0
    travel_time_total=0
    service_time_total=0
    route = ind2route(individual, instance)
    total_cost = 0
    time_windows = [
        (0.0, 60.0, 70),  # TW 0 AM 
        (61.0, 120.0, 65),# TW 1 AM 
        (121.0, 180.0, 65),# TW 2 AM 
        (181.0, 240.0, 70),# TW 3 AM 
        (241.0, 300.0, 65),# TW 4 AM 
        (301.0, 360.0, 60),# TW 5 AM 
        (361.0, 420.0, 55), # TW 6 AM 
        (421.0, 480.0, 50),#TW 7 AM 
        (481.0, 540.0, 50),#TW 8 AM 
        (541.0, 600.0, 45),#TW 9 AM 
        (601.0, 660.0, 40),#TW 10 AM 
        (661.0, 720.0, 30),#TW 11 AM 
        (721.0, 780.0, 30),#TW 12 AM 
        (781.0, 840.0, 30),#TW 1 PM 
        (841.0, 900.0, 35),#TW 2 PM 
        (901.0, 960.0, 40),#TW 3 PM 
        (961.0, 1020.0, 40),#TW 4 PM 
        (1021.0,1080.0, 30),#TW 5 PM 
        (1081.0,1140.0, 30),#TW 6 PM 
        (1141.0,1200.0, 45),#TW 7 PM 
        (1201.0,1260.0, 50),#TW 8 PM 
        (1261.0,1320.0, 60),#TW 9 PM 
        (1321.0,1380.0, 65),#TW 10 PM      
        (1381.0,1440.0, 70),#TW 11 PM 
    ]
    def speed_function(time):
        hour = time 
        for start, end, speed in time_windows:
            if start <= hour < end:
                return speed
        return 50
    for sub_route in route:
        sub_route_time_cost = 0
        sub_route_distance = 0
        elapsed_time = 0
        last_customer_id = 0
        for customer_id in sub_route:
            # Calculate section distance
            distance = instance['distance_matrix'][last_customer_id][customer_id]
            # Update sub-route distance
            sub_route_distance = sub_route_distance + distance
            # Calculate time cost
            speed = speed_function((instance[f'customer_{customer_id}']['ready_time']+(instance[f'customer_{customer_id}']['due_time']))/2)
            travel_time = distance /speed
            travel_time_total=travel_time_total+travel_time
            arrival_time = elapsed_time + travel_time 
            time_cost = wait_cost * max(instance[f'customer_{customer_id}']['ready_time'] - \
                arrival_time, 0) + delay_cost * max(arrival_time - \
                instance[f'customer_{customer_id}']['due_time'], 0)
            # Update sub-route time cost
            sub_route_time_cost = sub_route_time_cost + time_cost
            # Update elapsed time
            elapsed_time = arrival_time + instance[f'customer_{customer_id}']['service_time']
            service_time_total=service_time_total+ instance[f'customer_{customer_id}']['service_time']
            # Update last customer ID
            last_customer_id = customer_id
        # Calculate transport cost
        sub_route_distance = sub_route_distance + instance['distance_matrix'][last_customer_id][0]
        sub_route_transport_cost = init_cost + unit_cost * sub_route_distance
       #emission calcule
        #sub_route_emission_cost=sub_route_emission_cost+((a/speed+b+c*speed+d*speed**2+e*speed**3+f*speed**4+g*speed**5)*sub_route_distance)
        sub_route_emission_cost=sub_route_emission_cost+((k+l*g1+q*g1**2+r*speed+u/speed)*sub_route_distance) #* carbon_cost
        # Obtain sub-route cost
        sub_route_cost = sub_route_emission_cost #sub_route_time_cost+sub_route_transport_cost
        total_distance=total_distance+sub_route_distance
        # Update total cost
        total_cost = total_cost + sub_route_cost
    #ttdistance=1/total_distance    
    fitness = 1/total_cost
    #fitness = 1/total_distance
    return (fitness,total_distance)


def cx_partially_matched(ind1, ind2):
    cxpoint1, cxpoint2 = sorted(random.sample(range(min(len(ind1), len(ind2))), 2))
    part1 = ind2[cxpoint1:cxpoint2+1]
    part2 = ind1[cxpoint1:cxpoint2+1]
    rule1to2 = list(zip(part1, part2))
    is_fully_merged = False
    while not is_fully_merged:
        rule1to2, is_fully_merged = merge_rules(rules=rule1to2)
    rule2to1 = {rule[1]: rule[0] for rule in rule1to2}
    rule1to2 = dict(rule1to2)
    ind1 = [gene if gene not in part2 else rule2to1[gene] for gene in ind1[:cxpoint1]] + part2 + \
        [gene if gene not in part2 else rule2to1[gene] for gene in ind1[cxpoint2+1:]]
    ind2 = [gene if gene not in part1 else rule1to2[gene] for gene in ind2[:cxpoint1]] + part1 + \
        [gene if gene not in part1 else rule1to2[gene] for gene in ind2[cxpoint2+1:]]
    return ind1, ind2

def mut_shuffle_indexes(individual):
    start, stop = sorted(random.sample(range(len(individual)), 2))
    segment = individual[start:stop+1]
    random.shuffle(segment)
    individual[start:stop+1] = segment
    return (individual, )

def initialize_population(ind_size, pop_size):
    population = []
    for _ in range(pop_size):
        # Generate a random permutation
        individual = random.sample(range(1, ind_size + 1), ind_size)
        
        # Apply the sweep heuristic
        start = random.choice(individual)  # Choose a random starting point
        sorted_individual = [start]
        remaining = list(set(individual) - {start})
        
        while remaining:
            # Find the closest element to the last added element
            closest = min(remaining, key=lambda x: abs(x - sorted_individual[-1]))
            sorted_individual.append(closest)
            remaining.remove(closest)
        
        population.append(creator.Individual(sorted_individual))
    
    return population

def local_search(individual, toolbox, max_iters=100):
    best = toolbox.clone(individual)
    best_fitness, best_distance = toolbox.evaluate(best)  # Utiliser la valeur de fitness et la distance
    best.fitness.values = (best_fitness,)

    for _ in range(max_iters):
        neighbor = toolbox.clone(best)
        toolbox.mutate(neighbor)
        neighbor_fitness, neighbor_distance = toolbox.evaluate(neighbor)  # Utiliser la valeur de fitness et la distance
        neighbor.fitness.values = (neighbor_fitness,)

        if neighbor.fitness.values[0] > best.fitness.values[0]:
            best = neighbor
            best_fitness = neighbor_fitness
            best_distance = neighbor_distance

    return best, best_distance

import time

def run_gavrptw(carbon_cost, a, b, c, d, e, f, g, v, instance_name, unit_cost, init_cost, wait_cost, delay_cost, ind_size, pop_size, 
                cx_pb, mut_pb, n_gen, export_csv=False, customize_data=False):
    start_time = time.time()  # Début du chronomètre
    best_distance = None
    fitness_distance_data = []

    if customize_data:
        json_data_dir = os.path.join(BASE_DIR, 'data', 'json_customize')
    else:
        json_data_dir = os.path.join(BASE_DIR, 'data', 'json')
    json_file = os.path.join(json_data_dir, f'{instance_name}.json')
    instance = load_instance(json_file=json_file)
    if instance is None:
        return

    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    toolbox.register('indexes', random.sample, range(1, ind_size + 1), ind_size)
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', lambda: initialize_population(ind_size, pop_size))
    toolbox.register('evaluate', eval_vrptw, instance=instance, a=a, b=b, c=c, d=d, e=e, f=f, g=g, v=v, unit_cost=unit_cost, 
                     init_cost=init_cost, wait_cost=wait_cost, delay_cost=delay_cost)
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('mate', cx_partially_matched)
    toolbox.register('mutate', mut_shuffle_indexes)
    pop = toolbox.population()

    csv_data = []

    print('Start of evolution')
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, (fit, distance) in zip(pop, fitnesses):
        ind.fitness.values = (fit,)  # Utiliser seulement la valeur de fitness
        fitness_distance_data.append((fit, distance)) 
        print(f'Evaluated individual: {ind}, Fitness: {fit}, Distance: {distance}')
    print(f'  Evaluated {len(pop)} individuals')

    max_stagnation = 10
    stagnation_count = 0
    prev_best_fitness = None

    for gen in range(n_gen):
        print(f'-- Generation {gen} --')
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_pb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mut_pb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Apply local search to each offspring and update fitness_distance_data
        for i in range(len(offspring)):
            offspring[i], distance = local_search(offspring[i], toolbox)
            fitness_distance_data.append((offspring[i].fitness.values[0], distance))

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, (fit, distance) in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)  # Utiliser seulement la valeur de fitness
            fitness_distance_data.append((fit, distance)) 
            print(f'Evaluated individual: {ind}, Fitness: {fit}, Distance: {distance}')
        print(f'  Evaluated {len(invalid_ind)} individuals')

        pop[:] = offspring
        
        fits_and_distances = [(ind.fitness.values[0], distance) for ind, (_, distance) in zip(pop, fitness_distance_data)]
        fits = [fit for fit, _ in fits_and_distances]
        distances = [distance for _, distance in fits_and_distances]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum([x ** 2 for x in fits])
        std = abs(sum2 / length - mean ** 2) ** 0.5
        print(f'  Min {min(fits)}')
        print(f'  Max {max(fits)}')
        print(f'  Avg {mean}')
        print(f'  Std {std}')
        if export_csv:
            csv_row = {
                'generation': gen,
                'evaluated_individuals': len(invalid_ind),
                'min_fitness': min(fits),
                'max_fitness': max(fits),
                'avg_fitness': mean,
                'std_fitness': std,
            }
            csv_data.append(csv_row)

        current_best_fitness = max(fits)
        if prev_best_fitness is not None and current_best_fitness == prev_best_fitness:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best_fitness = current_best_fitness

        if stagnation_count >= max_stagnation:
            print(f'Stagnation detected. Stopping early at generation {gen}.')
            break

    print('-- End of (successful) evolution --')
    best_ind = tools.selBest(pop, 1)[0]
    best_fitness = best_ind.fitness.values[0]
    for fit, distance in fitness_distance_data:
        if fit == best_fitness:
            best_distance = distance 
    print(f'Best individual: {best_ind}')
    print(f'Fitness: {best_ind.fitness.values[0]}')
    print_route(ind2route(best_ind, instance))
    print(f'Total emissions in Grams: {1 / best_ind.fitness.values[0]}')
    print(f'Total distance in km: {best_distance}')

    end_time = time.time()  # Fin du chronomètre
    execution_time = end_time - start_time
    print(f'Total execution time: {execution_time:.2f} seconds')
