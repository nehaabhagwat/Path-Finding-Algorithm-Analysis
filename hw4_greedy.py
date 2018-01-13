# hw4_greedy.py
# Author: Neha Bhagwat
# Program solves TSP using nearest neighbour heuristic

import math
import time
import random


def calculate_distance(coord1, coord2, metric = "euclidean"):
    '''Function to find the euclidean distance between two co-ordinates'''
    if metric == "euclidean":
        x1, y1 = coord1
        x2, y2 = coord2
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)
        return(math.sqrt(math.pow((x1-x2),2) + math.pow((y1-y2),2)))
    else:
        print("Unknown distance metric encountered.")
        return 0


def find_nearest_neighbour(node, explored_list, list_of_nodes, coords):
    '''Function to find the nearest neighbour from a particular node'''
    distances = []
    for n in list_of_nodes:
        if n not in explored_list:
            distances.append((n, calculate_distance(coords[node], coords[n])))
    min_dist = 99999
    min_node = 99999
    for counter in range(0,len(distances)):
        n, dist = distances[counter]
        if dist < min_dist:
            min_dist = dist
            min_node = n

    return((min_node, min_dist))

def solve_TSP(list_of_nodes, coords):
    '''Function finds a path and tour length for a list of nodes and their co-ordinates'''
    start_node = random.randint(0,len(list_of_nodes))
    explored_list = []
    tour_length = 0
    explored_list.append(start_node)
    current_node = start_node
    while(len(explored_list)<len(list_of_nodes)):
        neighbour = find_nearest_neighbour(current_node, explored_list, list_of_nodes, coords)
        nn_node, nn_distance = neighbour
        explored_list.append(nn_node)
        tour_length += nn_distance
        current_node = nn_node
    # print("The length of the tour found using nearest neighbour heuristic and greedy algorithm is: " + str(round(tour_length,2)))
    return(explored_list, tour_length)

file_read = 0
# while loop to make sure that the filename entered is a valid file 
while file_read == 0:
    print("Enter the path of the file which contains the nodes and co-ordinates.")
    filename = raw_input()
    # input_file = open("att48.tsp", "r")
    try:
        input_file = open(filename, "r")
        file_read += 1
    except IOError:
        print("There was some problem in opening the file. Please enter a valid file path.")

# input_file = open("att48.tsp", "r")
input_data = input_file.readlines()
data_len = len(input_data)
# Remove the header lines from the data that is read
input_data = input_data[6:data_len]
coords = []
list_of_nodes = []
node_count = 0
# For loop to save the nodes or cities and their co-ordinates
for line in input_data:
    line = line.rstrip('\n')
    # print(line)
    data_list = line.split(' ')
    # print(data_list)
    list_of_nodes.append(node_count)
    node_count += 1
    coords.append((float(data_list[1]) , float(data_list[2])))
num_of_cities = len(coords)
start_time = time.time()
explored_list, tour_length = solve_TSP(list_of_nodes, coords)
# Add the distance from the final node back to the first node to the tour length
tour_length += calculate_distance(coords[explored_list[0]], coords[explored_list[len(explored_list)-1]])
# Add the first element back to the explored list
explored_list.append(explored_list[0])
end_time = time.time() - start_time
for i in range(0, len(explored_list)):
    explored_list[i] += 1


print("Path traversed: " + str(explored_list))
print("Tour Length: " + str(tour_length))
print("Time required for execution: " + str(end_time) + " seconds")

