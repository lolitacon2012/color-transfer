#get rid of the annoying round-to-int
from __future__ import division
__author__      = "Liu Dake"

import cv2
import numpy as np
from scipy.optimize import minimize, rosen, rosen_der
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
import math


#This part of code come from https://github.com/tarikd/python-kmeans-dominant-colors/blob/master/utils.py
#By Tarik Dadi
def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
 
    # return the histogram
    return hist

def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

#=====================parameters========================



#file names
reference_file = "fa.png"
target_file    = "q.png"
out_file    = "file.jpg"

#threshold of determining if this pixel belongs to white-points
t=0.2

#cluster numbers
target_cluster_number = 16
reference_cluster_number = 16


#Keep vertical_i_r == vertical_i_t and horizon_i_r == horizon_i_t if you don't know what you are doing

#vertical spacial sensitivity for reference
vertical_i_t = 40

#horizonal spacial sensitivity for reference
horizon_i_t = 40

#vertical spacial sensitivity for target
vertical_i_r = vertical_i_t

#horizonal-spacial sensitivity for reference
horizon_i_r = horizon_i_t

#weight for cluster size difference
percentage_space_ratio = 100

#weight for location size difference
space_difference_ratio = 50000

#weight for cluster size difference
color_difference_ratio = 50000

#step size of each white balance adjustment iteration
step = 0.8

#allowed error for white balance adjustment
allowed_error = 0.0007

#maxium iteration for white balance adjustment
iteration_time = 1

#Standard Chromatic Adaptation Transform CAT02 matrix
#Not Used
M = np.matrix([[0.7328, 0.4296, -0.1624], [-0.7036, 1.6975, 0.0061], [0.0030, 0.0136, 0.9834]])

#Standard Sobel operators
#Deleted
Sobel_x = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Sobel_y = np.matrix([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

#Sobel operator for gradient-preserving color transfer final step
Sobel_op_full = Sobel_x.T*Sobel_x+Sobel_y.T*Sobel_y
Sobel_lam = 1

#Diag matrix to adjust white balance
Diag = np.zeros(shape=(3,3))
Diag[0][0] = 999
Diag[1][1] = 999
Diag[2][2] = 999

#average rgv for further calculation
average_b_r = 0
average_g_r = 0
average_r_r = 0

average_b_t = 0
average_g_t = 0
average_r_t = 0



#===================Images======================#
# Read the images

reference_bgr = cv2.imread(reference_file)
target_bgr    = cv2.imread(target_file)

reference_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB)
target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)

original_reference_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB)
original_target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)

reference_xyz = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2XYZ)
target_xyz = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2XYZ)

target_bgr_origin = target_bgr

# Prepare size data
r_height, r_width, r_channels = reference_lab.shape
t_height, t_width, t_channels = target_lab.shape

#initialize
E_avg = np.zeros((1,3))
I_avg = np.zeros((1,3))
E_L_avg = 0
I_L_avg = 0

#Estimate white point for reference in BGR space
output_r_lab = reference_lab
reference_lab = reference_lab.astype(np.int64)
counter = 0
color_and_location_reference = np.zeros((reference_lab.shape[0] * reference_lab.shape[1], 5))
color_and_location_reference_counter = 0

for i in range(r_width):
        for j in range(r_height):
                color_and_location_reference[color_and_location_reference_counter][0] = j*horizon_i_r/r_width
                color_and_location_reference[color_and_location_reference_counter][1] = i*vertical_i_r/r_height
                color_and_location_reference[color_and_location_reference_counter][2] = reference_lab[j][i][0]
                color_and_location_reference[color_and_location_reference_counter][3] = reference_lab[j][i][1]
                color_and_location_reference[color_and_location_reference_counter][4] = reference_lab[j][i][2]

                color_and_location_reference_counter = color_and_location_reference_counter + 1
        	if abs(reference_lab[j][i][1]-128)+abs(reference_lab[j][i][2]-128)<t*reference_lab[j][i][0]/2.55:
        		#mark white-points as pink for output
                        output_r_lab[j][i][0] = 180
        		output_r_lab[j][i][1] = 250
        		output_r_lab[j][i][2] = 0
                        #sum of all white-points, BGR
        		E_avg[0][0] = E_avg[0][0] + reference_bgr[j][i][0]
        		E_avg[0][1] = E_avg[0][1] + reference_bgr[j][i][1]
        		E_avg[0][2] = E_avg[0][2] + reference_bgr[j][i][2]
                        #sum of illumination
                        E_L_avg = E_L_avg + reference_lab[j][i][0]
        		counter = counter + 1

#write the white-points
cv2.imwrite("whitepoint_reference.jpg", cv2.cvtColor(output_r_lab, cv2.COLOR_LAB2BGR))

#calculate average
if counter == 0:
    counter = 1
E_avg[0] = E_avg[0]/counter
E_L_avg = E_L_avg/counter
print "Average ilumination of reference:"
print E_L_avg

#Normalization. White-points form a straight line called white balance line, which defines the white without the influence of illumination.
#Thus, it should not be affected by the length, but only the components of color.
Sum_E = E_avg[0][0] + E_avg[0][1] + E_avg[0][2]
if Sum_E == 0:
    Sum_E = 1
E_avg[0] = E_avg[0]/Sum_E


target_lab = target_lab.astype(np.int64)
target_bgr = target_bgr.astype(np.int64)




#iteration
for it in range (0, iteration_time):
        
        target_lab = cv2.cvtColor(target_bgr.astype(np.uint8), cv2.COLOR_BGR2LAB)
        print "total white difference:"
        print abs(Diag[2][2]-1)+abs(Diag[1][1]-1)+abs(Diag[0][0]-1)

        if abs(Diag[2][2]-1)+abs(Diag[1][1]-1)+abs(Diag[0][0]-1)<allowed_error:
                print "converged. stopping now..."
                break

        print it
        output_t_lab = target_lab
        target_lab = target_lab.astype(np.int64)
        counter = 0
        color_and_location_target_counter = 0;
        color_and_location_target = np.zeros((target_lab.shape[0] * target_lab.shape[1], 5))
        for i in range(t_width):
                for j in range(t_height):
                        color_and_location_target[color_and_location_target_counter][0] = j*horizon_i_t/t_width
                        color_and_location_target[color_and_location_target_counter][1] = i*vertical_i_t/t_height
                        color_and_location_target[color_and_location_target_counter][2] = target_lab[j][i][0]
                        color_and_location_target[color_and_location_target_counter][3] = target_lab[j][i][1]
                        color_and_location_target[color_and_location_target_counter][4] = target_lab[j][i][2]
                        color_and_location_target_counter = color_and_location_target_counter + 1
                	if abs(target_lab[j][i][1]-128)+abs(target_lab[j][i][2]-128)<t*target_lab[j][i][0]/2.55:
                		#mark white-points for output
                                output_t_lab[j][i][0] = 200
                		output_t_lab[j][i][1] = 255
                		output_t_lab[j][i][2] = 0
                                #sum of all white-points, BGR
                		I_avg[0][0] = I_avg[0][0] + target_bgr[j][i][0]
                		I_avg[0][1] = I_avg[0][1] + target_bgr[j][i][1]
                		I_avg[0][2] = I_avg[0][2] + target_bgr[j][i][2]
                                #sum of illumination
                                I_L_avg = I_L_avg + target_lab[j][i][0]
                		counter = counter + 1

        cv2.imwrite(str(it)+"_whitepoint_target.jpg", cv2.cvtColor(output_t_lab, cv2.COLOR_LAB2BGR))

        #calculate average
        if counter<1 or iteration_time < 2:
            break;
                
        I_avg[0] = I_avg[0]/counter
        I_L_avg = I_L_avg/counter
        Sum_I = I_avg[0][0] + I_avg[0][1] + I_avg[0][2]
        I_avg[0] = I_avg[0]/Sum_I

        #diag matrix to adjust color.
        Diag[0][0] = 1 - step * (1 - E_avg[0][0]/(I_avg[0][0]))
        Diag[1][1] = 1 - step * (1 - E_avg[0][1]/(I_avg[0][1]))
        Diag[2][2] = 1 - step * (1 - E_avg[0][2]/(I_avg[0][2]))
        print Diag

        for i in range(t_width):
        	for j in range(t_height):
                        #change color
        		target_bgr[j][i] = np.array((Diag*(np.matrix(target_bgr[j][i]).T)).T)

                        #avoid overflow. Currently it uses the following naive algorithm.
                        #i will improve it later (adjust according to how white/dark it is)
                        for k in range (0, 3):
                                if target_bgr[j][i][k]>255:
                                        target_bgr[j][i][k] = 255
                                if target_bgr[j][i][k]<0:
                                        target_bgr[j][i][k] = 0
                
        cv2.imwrite("0"+str(it)+"_result.jpg", target_bgr)

#illumination adjustment
if I_L_avg == 0:
    E_L_avg = 1
    I_L_avg = 1

if iteration_time > 1:
    for i in range(t_width):
        for j in range(t_height):
                I_Mo = (1 + 0.7*((255 - target_lab[j][i][0])/255) *((E_L_avg/I_L_avg) -1 ))
                
                target_lab[j][i][0] =  I_Mo*target_lab[j][i][0]
                
target_bgr = cv2.cvtColor(target_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
target_bgr = target_bgr_origin

cv2.imwrite("Final_result.jpg", target_bgr)

#re-coloring
print "now starting re-coloring..."

#clustering
clt = KMeans(n_clusters = reference_cluster_number)
clt.fit_transform(color_and_location_reference)
labeled_areas = clt.labels_
#labled area for reference
reference_labled = labeled_areas

E_colors = np.zeros((reference_cluster_number,7))
I_colors = np.zeros((target_cluster_number,7))

reference_R_counters = np.zeros((reference_cluster_number,r_width*r_height))
reference_G_counters = np.zeros((reference_cluster_number,r_width*r_height))
reference_B_counters = np.zeros((reference_cluster_number,r_width*r_height))
target_R_counters = np.zeros((target_cluster_number,t_width*t_height))
target_G_counters = np.zeros((target_cluster_number,t_width*t_height))
target_B_counters = np.zeros((target_cluster_number,t_width*t_height))

image = reference_lab.astype(np.uint8)
counter = 0;
for j in range(r_width):
    for i in range(r_height):
        reference_B_counters[labeled_areas[counter]][counter] = image[i][j][0]
        reference_G_counters[labeled_areas[counter]][counter] = image[i][j][1]
        reference_R_counters[labeled_areas[counter]][counter] = image[i][j][2]
        counter = counter + 1;

hist = centroid_histogram(clt)
counter = 0
for (percent, color) in zip(hist, clt.cluster_centers_):
        E_colors[counter][0] = percent
        E_colors[counter][1] = color[2]
        E_colors[counter][2] = color[3]
        E_colors[counter][3] = color[4]
        E_colors[counter][4] = color[0]
        E_colors[counter][5] = color[1]
        E_colors[counter][6] = counter
        counter = counter + 1

clt = KMeans(n_clusters = target_cluster_number)
clt.fit_transform(color_and_location_target)
labeled_areas = clt.labels_

target_labled = labeled_areas
#labeled area for target

image = target_lab.astype(np.uint8)
counter = 0;
for j in range(t_width):
    for i in range(t_height):
        target_B_counters[labeled_areas[counter]][counter] = image[i][j][0]
        target_G_counters[labeled_areas[counter]][counter] = image[i][j][1]
        target_R_counters[labeled_areas[counter]][counter] = image[i][j][2]
        counter = counter + 1;

hist = centroid_histogram(clt)
counter = 0
for (percent, color) in zip(hist, clt.cluster_centers_):
        I_colors[counter][0] = percent
        I_colors[counter][1] = color[2]
        I_colors[counter][2] = color[3]
        I_colors[counter][3] = color[4]
        I_colors[counter][4] = color[0]
        I_colors[counter][5] = color[1]
        I_colors[counter][6] = counter
        counter = counter + 1

#sort the color table according to clustering label index
col = 6

I_colors_sorted = I_colors[np.argsort(I_colors[:,col])]
E_colors_sorted = E_colors[np.argsort(E_colors[:,col])]

print I_colors_sorted
print E_colors_sorted

mapping_table = np.zeros((target_cluster_number,2))
for k in range(target_cluster_number):
        current_selection = 0
        current_difference = 999999999
        for t in range(reference_cluster_number):
                vector_e = E_colors_sorted[t]
                #decending order

                color_difference = color_difference_ratio*np.linalg.norm(I_colors_sorted[k][1:4]-E_colors_sorted[t][1:4])/math.pow(256*256*3 ,0.5)
                percentage_difference = percentage_space_ratio*(I_colors_sorted[k][0]-E_colors_sorted[t][0])
                space_difference = space_difference_ratio*np.linalg.norm(I_colors_sorted[k][4:6]-E_colors_sorted[t][4:6])/math.pow(vertical_i_t*vertical_i_t+horizon_i_t*horizon_i_t  ,0.5)
                total_difference = abs(percentage_difference) + abs(color_difference) + abs(space_difference)
                
                if total_difference<current_difference and total_difference<999999999:
                        current_selection = E_colors_sorted[t][6]
                        current_difference = total_difference

                print("----------------------------------------------------")
                print("From %d" % k)
                print("To %d" % t)
                print("Color difference = %d" % abs(color_difference))
                print("Distance difference = %d" % abs(space_difference))
                print("Percentage difference = %d" % abs(percentage_difference))
        E_colors_sorted[current_selection][0] = E_colors_sorted[current_selection][0] + 999999999999
        mapping_table[k][0] = I_colors_sorted[k][6]
        mapping_table[k][1] = current_selection


col = 0
mapping_table = mapping_table[np.argsort(mapping_table[:,col])]


counter = 0
for i in range(t_width):
        for j in range(t_height):
                classification = target_labled[counter]
                mapped_index = mapping_table[classification][1]
                Sum_A = 0
                Sum_B = 0
                Sum_D = 0
                for region in range(target_cluster_number):
                    current_color = np.zeros(5)
                    current_color[3] = j*horizon_i_t/t_width
                    current_color[4] = i*vertical_i_t/t_height
                    current_color[0] = target_lab[j][i][0]
                    current_color[1] = target_lab[j][i][1]
                    current_color[2] = target_lab[j][i][2]
                    
                    current_distance = np.linalg.norm(E_colors_sorted[region][1:6] - current_color)
                    #This distance is calculated in the same ways as K-Means, in order to determine the influence from neighbour colors
                    
                    if current_distance == 0:
                        Sum_D = 1
                        Sum_A = E_colors_sorted[mapped_index][2]
                        Sum_B = E_colors_sorted[mapped_index][3]
                        break
                    else:
                        
                        """
                        Sum_L = Sum_L + ((((((E_colors_sorted[region][1]/current_distance)/current_distance)/current_distance)/current_distance))/current_distance)/current_distance
                        Sum_A = Sum_A + ((((((E_colors_sorted[region][2]/current_distance)/current_distance)/current_distance)/current_distance))/current_distance)/current_distance
                        Sum_B = Sum_B + ((((((E_colors_sorted[region][3]/current_distance)/current_distance)/current_distance)/current_distance))/current_distance)/current_distance
                        Sum_D = Sum_D + ((((((1/current_distance)/current_distance)/current_distance)/current_distance)/current_distance)/current_distance)
                        """
                        
                        #Exponential kernel
                        Sum_A = Sum_A + (E_colors_sorted[region][2]/math.pow(2, current_distance))
                        Sum_B = Sum_B + (E_colors_sorted[region][3]/math.pow(2, current_distance))
                        Sum_D = Sum_D + (1/math.pow(2, current_distance))

                        
                Sum_A = Sum_A / Sum_D
                Sum_B = Sum_B / Sum_D
                
                target_lab[j][i][1] = Sum_A
                target_lab[j][i][2] = Sum_B

                counter = counter + 1

final_bgr = cv2.cvtColor(target_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
cv2.imwrite(out_file, final_bgr)
print mapping_table