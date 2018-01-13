# hw4_wrapper.py
# Author: Neha Bhagwat

import os
import math
import re
import time
from pyevolve import G1DList, GAllele
from pyevolve import GSimpleGA
from pyevolve import Mutators
from pyevolve import Crossovers
from pyevolve import Consts
import sys, random
random.seed(1024)

PIL_SUPPORT = None

try:
   from PIL import Image, ImageFont, ImageDraw
   PIL_SUPPORT = True
except:
   PIL_SUPPORT = False

cm     = []
coords = []
CITIES = 48
WIDTH   = 9999
HEIGHT  = 9999
LAST_SCORE = -1
GREEDY_LENGTH = 0
GREEDY_TIME = 0
GA_LENGTH = 0
GA_TIME = 0
result = []
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

class genetic_algorithm_implementation:
    def implement_ga(self, coords):
        cm = self.cartesian_matrix(coords)
        genome = G1DList.G1DList(len(coords))
        genome.evaluator.set(lambda chromosome: self.tour_length(cm, chromosome))
        genome.crossover.set(Crossovers.G1DListCrossoverEdge)
        genome.initializator.set(self.G1DListTSPInitializator)

        ga = GSimpleGA.GSimpleGA(genome)
        ga.setGenerations(1000)
        ga.setMinimax(Consts.minimaxType["minimize"])
        ga.setCrossoverRate(1.0)
        ga.setMutationRate(0.02)
        ga.setPopulationSize(80)

        ga.evolve(freq_stats=200)
        best = ga.bestIndividual()
        iter_list = best.getInternalList()
        GA_LENGTH = self.tour_length(cm, best)
        print("Total length of the tour: " + str(GA_LENGTH))

    def cartesian_matrix(self, coords):
       """ A distance matrix """
       matrix={}
       for i,(x1,y1) in enumerate(coords):
          for j,(x2,y2) in enumerate(coords):
             dx, dy = x1-x2, y1-y2
             dist=math.sqrt(dx*dx + dy*dy)
             matrix[i,j] = dist
       return matrix

    def tour_length(self, matrix, tour):
       """ Returns the total length of the tour """
       total = 0
       t = tour.getInternalList()
       for i in range(CITIES):
          j      = (i+1)%CITIES
          total += matrix[t[i], t[j]]
       total += matrix[t[0], t[len(t)-1]]
       return total

    def write_tour_to_img(self, coords, tour, img_file):
       """ The function to plot the graph """
       padding=20
       coords=[(x+padding,y+padding) for (x,y) in coords]
       maxx,maxy=0,0
       # print("\n\nTOUR: " + str(tour))
       for x,y in coords:
          maxx, maxy = max(x,maxx), max(y,maxy)
       maxx+=padding
       maxy+=padding
       img=Image.new("RGB",(int(maxx),int(maxy)),color=(255,255,255))
       font=ImageFont.load_default()
       d=ImageDraw.Draw(img);
       num_cities=len(tour)
       for i in range(num_cities):
          j=(i+1)%num_cities
          city_i=tour[i]
          city_j=tour[j]
          x1,y1=coords[city_i]
          x2,y2=coords[city_j]
          d.line((int(x1),int(y1),int(x2),int(y2)),fill=(0,0,0))
          d.text((int(x1)+7,int(y1)-5),str(i),font=font,fill=(32,32,32))

       for x,y in coords:
          x,y=int(x),int(y)
          d.ellipse((x-5,y-5,x+5,y+5),outline=(0,0,0),fill=(196,196,196))
       del d
       img.save(img_file, "PNG")
       print "The plot was saved into the %s file." % (img_file,)

    def G1DListTSPInitializator(self, genome, **args):
       """ The initializator for the TSP """
       lst = [i for i in xrange(genome.getListSize())]
       random.shuffle(lst)
       genome.setInternalList(lst)

    
class greedy_nearest_neighbour:
    def find_nearest_neighbour(self, node, explored_list, list_of_nodes, edges):
        distances = []
        for n in list_of_nodes:
            if n not in explored_list:
                distances.append((n, edges[node][n]))
        min_dist = 99999999
        min_node = 99999999
        for counter in range(0,len(distances)):
            n, dist = distances[counter]
            if dist < min_dist:
                min_dist = dist
                min_node = n

        return((min_node, min_dist))

    def solve_TSP(self, list_of_nodes, edges):
        start_node = list_of_nodes[0]
        explored_list = []
        tour_length = 0
        explored_list.append(start_node)
        current_node = start_node
        while(len(explored_list)<len(list_of_nodes)):
            neighbour = self.find_nearest_neighbour(current_node, explored_list, list_of_nodes, edges)
            nn_node, nn_distance = neighbour
            explored_list.append(nn_node)
            # print(len(explored_list))
            tour_length += nn_distance
            current_node = nn_node
        # print("*************************************************")
        GREEDY_LENGTH = round(tour_length,2)
        # print("The length of the tour found using nearest neighbour heuristic and greedy algorithm is: " + str(round(tour_length,2)))
        return(explored_list)

print("Enter the path where the benchmark files are located.")
folder_path = raw_input()
# C:\Users\bhagw\Desktop\SJSU - SEM I\Topics_in_AI\Homework Documents\Assignment4\benchmarks
# C:\Users\bhagw\Desktop\SJSU - SEM I\Topics_in_AI\Homework Documents\Assignment4\2_benchmarks
list_of_tsp_files = []
for path, subdirs, files in os.walk(folder_path):
    for filename in files:
        f = os.path.join(path, filename)
        ext = os.path.splitext(filename)[1]
        if ext.lower().find("tsp") != -1:
            print("File detected: " + str(filename))
            list_of_tsp_files.append(f)
# print(list_of_tsp_files)
list_of_indices = []
output_file = open("results.csv", "w")
for i in range(0, len(list_of_tsp_files)):
    input_file = open(list_of_tsp_files[i], "r")
    for j in range(0, 20):
        header = input_file.readline()
        j += 1
        if (header.find("EDGE_WEIGHT_TYPE")!=-1) and (header.find("EUC_2D") != -1):
            input_data = input_file.readlines()
            data_len = len(input_data)
            if input_data[data_len - 1].find("EOF") == -1:
                input_data = input_data[1:data_len]
            else:
                input_data = input_data[1:data_len - 1]
                data_len = data_len - 1
            coords = []
            list_of_nodes = []
            
            node_count = 0
            # print(input_data)
            for line in input_data:
                line.rstrip('\n')
                # print(line)
                line = re.sub(' +', ' ', line)
                data_list = line.split(' ')
                # print(data_list)
                list_of_nodes.append(node_count)
                node_count += 1
                coords.append((float(data_list[1]) , float(data_list[2])))
            num_of_cities = len(coords)
            CITIES = num_of_cities
            edges = []
            for cities_1 in range(0, num_of_cities):
                temp_edges = []
                for cities_2 in range(0, num_of_cities):
                    temp_edges.append(calculate_distance(coords[cities_1], coords[cities_2], "euclidean"))
                edges.append(temp_edges)
            print(list_of_tsp_files[i])
            print("GREEEDY: ")
            greedy_obj = greedy_nearest_neighbour()
            start_time = time.time()
            explored_list = greedy_obj.solve_TSP(list_of_nodes, edges)
            explored_list.append(explored_list[0])
            GREEDY_LENGTH += calculate_distance(coords[explored_list[0]], coords[explored_list[len(explored_list)-2]])
            GREEDY_TIME = time.time() - start_time
            for counter in range(0, len(explored_list)):
                explored_list[counter] += 1
            # print(explored_list)
            print("Tour Length: " + str(GREEDY_LENGTH))
            print("Time required for execution: " + str(GREEDY_TIME) + " seconds")
            print("\nGENETIC ALGORITHM: ")
            start_time = time.time()
            
            ga_obj = genetic_algorithm_implementation()
            ga_obj.implement_ga(coords)
            GA_TIME = time.time() - start_time
            print("\n\n")
            result.append([GA_LENGTH, GA_TIME, GREEDY_LENGTH, GREEDY_TIME])
            
print(result)
            
                        
