#Based on Sentdex's tutorials

import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq
from statistics import mean
    
#Get the two major road lanes from the lines given
def get_lanes(lines):
    try:
        #Find the largest y value for the line marker, depending on where the horizon is
        y_vals = []
        for line in lines:
            for i in line:
                y_vals += [i[1], i[3]]

        min_y =  min(y_vals)
        max_y = 600
        new_lines = []
        line_dict = {}

        #Find all line equations
        for idx, i in enumerate(lines):
            for xyxy in i:
                #Find the least squares fit of the line
                x_coords = [xyxy[0], xyxy[2]]
                y_coords = [xyxy[1], xyxy[3]]
                A = vstack([x_coords, ones(len(x_coords))]).T
                m, b = lstsq(A, y_coords)[0]

                x1 = (min_y-b)/m
                x2 = (max_y-b)/m

                l = [int(x1), min_y, int(x2), max_y]
                line_dict[idx] = [m, b, l]
                new_lines.append(l)
                final_lanes = {}

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]

            if len(final_lanes) == 0:
                final_lanes[m] = [[m, b, lines]]
            else:
                found_copy = False
                for other_slp in final_lanes_copy:
                    if not found_copy:
                        if abs(1.2*other_slp) > abs(m) > abs(0.8*other_slp):
                            if abs(final_lanes_copy[other_slp][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_slp][0][1]*0.8):
                                final_lanes[other_slp].append([m,b,line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [[m, b, line]]

        #Find the most frequently occuring lines to be the road lanes
        line_counter = {}
        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]
        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        #Find the average lane
        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s))
                                                                        
        #Return the 2 new lines
        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2], lane1_id, lane2_id
 
    except Exception as e:
        print(str(e))
    
