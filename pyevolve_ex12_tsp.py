# pyevolve_ex12_tsp.py
# Author: Neha Bhagwat
# Program to implement GA on the ATT48.tsp file and print the tour length, time and path traversed


from pyevolve import G1DList, GAllele
from pyevolve import GSimpleGA
from pyevolve import Mutators
from pyevolve import Crossovers
from pyevolve import Consts
import time

import sys, random
random.seed(1024)
from math import sqrt

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

def cartesian_matrix(coords):
   """ A distance matrix """
   matrix={}
   for i,(x1,y1) in enumerate(coords):
      for j,(x2,y2) in enumerate(coords):
         dx, dy = x1-x2, y1-y2
         dist=sqrt(dx*dx + dy*dy)
         matrix[i,j] = dist
   return matrix

def tour_length(matrix, tour):
   """ Returns the total length of the tour """
   total = 0
   t = tour.getInternalList()
   # print t
   for i in range(len(t)-1):
      j      = (i+1)%(CITIES)
      # print t[i]
      # print t[j]
      total += matrix[t[i], t[j]]
   total = total + matrix[t[0], t[len(t)-1]]
   return total

def write_tour_to_img(coords, tour, img_file):
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

def G1DListTSPInitializator(genome, **args):
   """ The initializator for the TSP """
   lst = [i for i in xrange(genome.getListSize())]
   random.shuffle(lst)
   genome.setInternalList(lst)

def main_run():
   global cm, coords, WIDTH, HEIGHT
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
    
   # input_file = open("att48.tsp", "r")
   input_data = input_file.readlines()
   # print(input_data)
   data_len = len(input_data)
   input_data = input_data[6:data_len]
   CITIES= len(input_data)
   input_file.close()
   coords = []
   for line in input_data:
      line = line.rstrip('\n')
      # print(line)
      data_list = line.split(' ')
      # print(data_list)
      coords.append((float(data_list[1]) , float(data_list[2])))
   cm     = cartesian_matrix(coords)
   genome = G1DList.G1DList(len(coords))

   genome.evaluator.set(lambda chromosome: tour_length(cm, chromosome))
   genome.crossover.set(Crossovers.G1DListCrossoverEdge)
   genome.initializator.set(G1DListTSPInitializator)

   ga = GSimpleGA.GSimpleGA(genome)
   ga.setGenerations(1000)
   ga.setMinimax(Consts.minimaxType["minimize"])
   ga.setCrossoverRate(1.0)
   ga.setMutationRate(0.02)
   ga.setPopulationSize(80)

   ga.evolve(freq_stats=500)
   best = ga.bestIndividual()
   output_file = open("TSP_output.txt", "w")
   output_file.write("TOUR_SECTION\n")
   iter_list = best.getInternalList()
   iter_list.append(iter_list[0])
   print("Total length of the tour: " + str(tour_length(cm, best)))
   for i in range(len(iter_list)):
      iter_list[i] = iter_list[i] + 1
      output_file.write(str(iter_list[i]) + "\n")  
   print("best\n"+str(iter_list))
   # print(type(best))
   output_file.close()
   
   if PIL_SUPPORT:
      write_tour_to_img(coords, best, "tsp_result.png")
   else:
      print "No PIL detected, cannot plot the graph !"

if __name__ == "__main__":
   main_run()
