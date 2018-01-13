# hw4_astar.py
# Author: Neha Bhagwat
# Purpose: Program to solve TSP using A* algorithm
# To be improved:
# 1. Add 1 to each element of the final path before printing - Done
# 2. Improve the pass0 function
# 3. Divide the code into modules
# 4. Remove generation of separate edges and edges_array


import heapq
import copy
import sys
import math
import time

def calculate_distance(coord1, coord2, metric = "euclidean"):
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

class shortest_path:
    def __init__(self, num_of_nodes, list_of_nodes, edges_array, start_node, destination_node):
        self.num_of_nodes = num_of_nodes
        self.edges_array = edges_array
        self.start_node = start_node
        self.destination_node = destination_node
        self.list_of_nodes = list_of_nodes

    def find_shortest_path(self, algorithm = "djikstras"):
        if algorithm == "djikstra":
            visited = []
            distance_to_node = []
            for i in range(0, self.num_of_nodes):
                distance_to_node.append(99999)
            current_node = self.start_node
            visited.append(current_node)
            distance_to_node[current_node] = 0
            while(len(visited)<self.num_of_nodes):
                for node in self.list_of_nodes:
                    if node not in visited:
                        edge_dist = self.edges_array[current_node][node]
                        if edge_dist < distance_to_node[node]:
                            distance_to_node[node] = edge_dist
                min_dist_index = distance_to_node.index(min(distance_to_node))
                current_node = self.list_of_nodes[min_dist_index]
                visited.append(current_node)
            return distance_to_node[self.destination_node]
                
        else:
            print("Unknown algorithm. Returning 0")
            return 0

class MST:
    def __init__(self):
        self.parent = dict()
        self.rank = dict()

    def create_set(self, vertex):
        self.parent[vertex] = vertex
        self.rank[vertex] = 0

    def find(self, vertex):
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])
        return self.parent[vertex]

    def union(self, vertex1, vertex2):
        root1 = self.find(vertex1)
        root2 = self.find(vertex2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2
        if self.rank[root1] == self.rank[root2]:
            self.rank[root2] += 1

    def kruskal(self, graph):
        cost = 0
        for vertex in graph['vertices']:
            self.create_set(vertex)
            minimum_spanning_tree = set()
            edges = list(graph['edges'])
            edges.sort()
            #print edges
        for edge in edges:
            weight, vertex1, vertex2 = edge
            if self.find(vertex1) != self.find(vertex2):
                self.union(vertex1, vertex2)
                minimum_spanning_tree.add(edge)
                cost += weight
        return cost

def find_nearest_neighbour(node, closed_list, list_of_nodes, coords):
    distances = []
    
    for n in list_of_nodes:
        if n not in closed_list and n != node:
            distances.append((n, calculate_distance(coords[node], coords[n])))
    min_dist = 99999
    min_node = 99999
    for counter in range(0,len(distances)):
        n, dist = distances[counter]
        if dist < min_dist:
            min_dist = dist
            min_node = n

    return((min_node, min_dist))

def calculate_shortest_path(start_node, destination_node, list_of_nodes, num_of_nodes, edges_array):
    shortest_path_object = shortest_path(num_of_cities, list_of_nodes, edges_array, start_node, destination_node)
    return(shortest_path_object.find_shortest_path("djikstra"))

def find_list_of_shortest_paths(start_node, list_of_nodes, num_of_nodes, edges_array):
    shortest_distances = []
    for nd in list_of_nodes:
        shortest_distances.append(calculate_shortest_path(nd, start_node, list_of_nodes, num_of_cities, edges_array))
    return shortest_distances

def pass0(num_of_cities, edges_array, edges, list_of_nodes, coords):
    # For pass0, g(node) will be 0 for all nodes
    # Following section calculates the h(node)
    h_values = []
    closed_list = []
    
    for node in list_of_nodes:
        shortest_distances = find_list_of_shortest_paths(node, list_of_nodes, num_of_cities, edges_array)
        print(node)
        closed_list = [node]
        nearest_neighbour, nn_distance = find_nearest_neighbour(node, closed_list, list_of_nodes, coords)
        
        unexplored_edges = []
        for t in edges:
            start, end, dist = t
            if (start != node) and (end!=node):
                unexplored_edges.append((dist, start, end))
        g = {
                    'vertices': list_of_nodes,
                    'edges': set(unexplored_edges)
                    }
        spanning_tree = MST()
        MST_cost = spanning_tree.kruskal(graph)
        min_path = min(shortest_distances[0:node] + shortest_distances[node+1:])
        
        h_values.append(nn_distance + MST_cost + min_path)
        closed_list = []
    print(h_values)
    print(h_values.index(min(h_values)))
    return (h_values.index(min(h_values)), h_values)

def implement_astar(pass0_tuple, num_of_cities, edges_array, edges, list_of_nodes, coords):
    start_node, f_values = pass0_tuple
    shortest_distances = find_list_of_shortest_paths(start_node, list_of_nodes, num_of_cities, edges_array)
    pq = []
    for c in range(0,len(list_of_nodes)):
        pq.append([f_values[c], list_of_nodes[c], [str(list_of_nodes[c])], 0, 0])
    heapq.heapify(pq)
    print("l1: " + str(len(list(pq))))
    closed_list = []
    it = 0
    while(len(closed_list)<num_of_cities):
        it += 1
        if (num_of_cities - len(closed_list) == 1):
            last_node = 99999
            for node in list_of_nodes:
                if node not in closed_list:
                    last_node = node
            closed_list.append(last_node)
            print(closed_list)
            break
        else:
            least_ele = heapq.heappop(pq)
            print("l2: " + str(len(list(pq))))
            f_val, current_node, traversed_list, g_val, iter_num = least_ele
            del_ele = []
            # If I go back, do I have to delete the nodes discovered so far?
            # The following section of code will remove the nodes discovered after the iteration to which you go back if you encounter a smaller f value
            # *********************************************************************************
            for i in range(0, len(pq)):
                l, m, n, o, p = pq[i]
                if p > iter_num:
                    del_ele.append(i)
            for i in sorted(del_ele, reverse=True):
                pq[i] = pq[-1]
                pq.pop()
                if i < len(pq):
                    heapq._siftup(pq, i)
                    heapq._siftdown(pq, 0, i)
            # *********************************************************************************   
            closed_list = []
            for node in traversed_list:
                closed_list.append(int(node))
            # closed_list.append(current_node)
            unexplored = []
            for node in list_of_nodes:
                if node not in closed_list:
                    unexplored.append(node)
            for node in list_of_nodes:
                if node not in closed_list:
                    print("closed_list: " + str(closed_list))
                    print("iter_num: " + str(iter_num))
                    print("node: " + str(node))
                    node_g = g_val + edges_array[current_node][node]
                    nearest_neighbour , h1 = find_nearest_neighbour(node, closed_list, list_of_nodes, coords)
                    g = MST()
                    unexplored_edges = []
                    for count1 in unexplored:
                        for count2 in unexplored:
                            if (count1 != count2) and (count1 != node) and (count2!=node):
                                unexplored_edges.append((edges_array[count1][count2], count1, count2))
                    g = {
                    'vertices': list_of_nodes,
                    'edges': set(unexplored_edges)
                    }
                    spanning_tree = MST()
                    h2 = spanning_tree.kruskal(g)
                    h3 = min(shortest_distances[0:node] + shortest_distances[node+1:])
                    t_list = traversed_list + [str(node)]
                    print("traversed_list: " + str(traversed_list))
                    print("node: " + str(node))
                    print("t_list: " + str(t_list))
                    heapq.heappush(pq, [node_g + h1 + h2 + h3, node, t_list, node_g, it])
            print(closed_list)
    return closed_list
                
file_read = 0
while file_read == 0:
    print("Enter the path of the file which contains the nodes and co-ordinates.")
    filename = raw_input()
    # input_file = open("att48.tsp", "r")
    try:
        input_file = open(filename, "r")
        file_read += 1
    except IOError:
        print("There was some problem in opening the file. Please enter a valid file path.")
    
start_time = time.time()
input_data = input_file.readlines()
data_len = len(input_data)

input_data = input_data[6:data_len]
coords = []
list_of_nodes = []
node_count = 0
print("saving input data in a list")
for line in input_data:
    line = line.rstrip('\n')
    # print(line)
    data_list = line.split(' ')
    # print(data_list)
    list_of_nodes.append(node_count)
    node_count += 1
    datalist = []
    for data in data_list:
        if data !='':
            datalist.append(data)
    x = datalist[1].strip(" ")
    # print "x" + str(x)
    y = datalist[2].rstrip(" ")
    coords.append((float(x) , float(y)))
num_of_cities = len(coords)
edges_array = []
print("creating an array of edges")
for count_start in range(0, num_of_cities):
    temp_list = []
    for count_end in range(0, num_of_cities):
        temp_list.append(calculate_distance(coords[count_start], coords[count_end]))
    edges_array.append(temp_list)
# print(edges_array)
print("creating edges")
edges = []
for count_start in range(0, num_of_cities):
    for count_end in range(0, num_of_cities):
        if count_start != count_end:
            edges.append((edges_array[count_start][count_end], count_start, count_end))

print("creating a graph")          
graph = {
'vertices': list_of_nodes,
'edges': set(edges)
}
print("implementing the A* algorithm")
pass0_tuple = pass0(num_of_cities, edges_array, edges, list_of_nodes, coords)
path = implement_astar(pass0_tuple, num_of_cities, edges_array, edges, list_of_nodes, coords)
dist = 0
for index in range(0,len(path)-1):
    dist = dist + edges_array[path[index]][path[index+1]]
dist = dist + edges_array[path[0]][path[len(path)-1]]
path.append(path[0])
# print(dist)
end_time = time.time() - start_time
for index in range(0, len(path)):
    path[index] += 1
print("Path traversed: " + str(path))
print("Tour Length: " + str(dist))
print("Time required for execution: " + str(end_time) + " seconds.")

