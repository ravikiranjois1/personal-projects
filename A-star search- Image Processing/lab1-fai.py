__author = 'RJ'

'''
A program to go from point A to point B on a map of the Mendon Ponds Park.
It involves scanning an image for the pixels and creating nodes on the required ones according to the input file.
Then, these nodes are travelled across using A* search to find the shortest distance between the start node and the
goal node

@author: Ravikiran Jois Yedur Prabhakar 
'''

import math
import time

from PIL import Image
import sys
from Node import Node
from queue import PriorityQueue

total_distance = 0
total_time = 0


def read_image(image_of_terrain):
    '''
    Method to read the input image file
    :param image_of_terrain: image file
    :return:loaded image file
    '''
    image = Image.open(image_of_terrain).load()
    return image


def read_elevation_txt(elevation_file):
    '''
    Method to read the elevation text file
    :param elevation_file: elevation file
    :return: 2D list of elevation information
    '''
    row_in_file = []
    list_of_rows = []
    with open(elevation_file, 'r') as f:
        list_of_rows = []
        for row in f:
            new_row_with_nums = []
            row_in_file = row.split()
            for item in row_in_file:
                new_row_with_nums.append(float(item[:item.index('e')])*100)
            list_of_rows.append(new_row_with_nums)
    return list_of_rows


def fetch_terrain_image_content(image_of_terrain, current_node, go_to):
    '''
    To get the pixels of the image
    :param image_of_terrain: input image
    :param current_node: source node
    :param go_to: desination node
    :return: pixels of current node and next node
    '''
    # print(image_of_terrain[current_node.x_coord])
    current_node_terrain = image_of_terrain[current_node[0], current_node[1]]
    # print(current_node_terrain)
    next_node_terrain = image_of_terrain[go_to.x_coord, go_to.y_coord]
    return current_node_terrain[:-1], next_node_terrain[:-1]


def fetch_speed_content(current_node_data, go_to_node_data, season):
    '''
    To get the speed that can be travelled across each pixel
    :param current_node_data: the current node
    :param go_to_node_data: the next node
    :param season: season
    :return: speed of current node and next node
    '''
    speed = dict()
    speed[(248, 148, 18)] = 100
    speed[(255, 192, 0)] = 75
    speed[(255, 255, 255)] = 65
    speed[(2, 208, 60)] = 50
    speed[(2, 136, 40)] = 45
    speed[(5, 73, 24)] = 0.1
    speed[(0, 0, 255)] = 5
    speed[(71, 51, 3)] = 150
    speed[(0, 0, 0)] = 95
    speed[(205, 0, 101)] = 0.1
    if season == 'fall':
        speed[(255, 255, 255)] = 20
    if season == 'winter':
        speed[(170, 200, 255)] = 10
    if season == 'spring':
        speed[(114, 104, 30)] = 15
    return speed[current_node_data], speed[go_to_node_data]


def fetch_elevation_content(elevation_file, current_node, go_to):
    '''
    To get the elevation content
    :param elevation_file: The elevation file data
    :param current_node: the current node data
    :param go_to: the next node data
    :return: the difference in elevation between next node and current node
    '''
    # print("Elevation:",elevation_file[current_node[1]][current_node[0]])
    current_node_elevation = elevation_file[current_node[1]][current_node[0]]
    next_node_elevation = elevation_file[go_to.y_coord][go_to.x_coord]
    return next_node_elevation - current_node_elevation


def dist_between_two_points(elevation, way):
    '''
    The distance between two pixels
    :param elevation: elevation difference between two pixels
    :param way: direction to travel
    :return: slope information and the distance
    '''
    global total_distance
    if way == 'left' or way == 'right':
        distance = math.sqrt(7.55**2 + elevation**2)
        # distance = 7.55
    elif way == 'top' or way == 'bottom':
        distance = math.sqrt(10.29**2 + elevation**2)
        # distance = 10.29
    elif way == 'top-left' or way == 'bottom-left' or way == 'top-right' or way == 'bottom-right':
        distance = math.sqrt(10.29**2 + 7.55**2)
    valley = (elevation/distance)*100
    # global total_distance
    total_distance += distance
    return valley, distance


def calculating_heuristic_value(go_to, goal_node):
    '''
    Calculating the heuristic value from node to goal node
    :param go_to: next node
    :param goal_node: target node
    :return: heuristic value of the next node
    '''
    hueristic_val = math.sqrt((goal_node.x_coord - go_to.x_coord)**2 + (goal_node.y_coord - go_to.y_coord)**2)
    return hueristic_val


def calculate_time_taken(valley, distance, current_node_speed, go_to_node_speed, current_node, go_to, cost_till_now, goal_node):
    '''
    To calculate the time cost between two pixels
    :param valley: slope
    :param distance: distance between two pixels
    :param current_node_speed: current pixel speed
    :param go_to_node_speed: next pixel speed
    :param current_node: current node
    :param go_to: next node
    :param cost_till_now: cost dictionary
    :param goal_node: target node
    :return: time taken/cost
    '''
    global total_distance
    if current_node_speed == 0 or go_to_node_speed == 0:
        time_to_pass = 1000000000000000000000000000
    else:
        half_dist = distance/2
        time_to_pass = (half_dist / current_node_speed) + (half_dist / go_to_node_speed)
    if valley < 0:
        time_to_pass -= distance * valley
    else:
        time_to_pass += distance * valley
    go_to.time_taken = cost_till_now[(current_node[0], current_node[1])] + time_to_pass
    heuristic_value = calculating_heuristic_value(go_to, goal_node)
    time_taken = go_to.time_taken + heuristic_value
    cost_till_now[(go_to.x_coord, go_to.y_coord)] = go_to.time_taken
    return time_taken


def a_star_search(start_node, goal_node, image_of_terrain, elevation_file, season):
    '''
    A* search to find cost between two points in the image
    :param start_node: start node
    :param goal_node: end node
    :param image_of_terrain: input image
    :param elevation_file: elevation information
    :param season: season
    :return: cost and parent path dictionary
    '''
    frontier = PriorityQueue()
    frontier.put((0, (start_node.x_coord, start_node.y_coord)))
    parent_path = {}
    parent_path[(start_node.x_coord, start_node.y_coord)] = None
    cost_till_now = {}
    cost_till_now[(start_node.x_coord, start_node.y_coord)] = 0

    while not frontier.empty():
        current_node = frontier.get()[1]
        neighbours = [[-1, 0, 'left'], [-1, -1, 'top-left'], [-1, 1, 'bottom-left'],
                      [0, -1, 'top'], [0, 1, 'bottom'],
                      [1, 0, 'right'], [1, -1, 'top-right'], [1, 1, 'bottom-right']]
        # print(current_node.y_coord, current_node.y_coord, goal_node.x_coord, goal_node.y_coord)
        if current_node[0] == goal_node.x_coord and current_node[1] == goal_node.y_coord:
            return cost_till_now[(goal_node.x_coord, goal_node.y_coord)], parent_path

        for neighbour in neighbours:
            check_this_neighbour = (neighbour[0] + current_node[0],
                                    neighbour[1] + current_node[1])
            x_coordinate, y_coordinate = check_this_neighbour[0], check_this_neighbour[1]
            way = neighbour[2]
            # print(x_coord, y_coord)
            if 0 <= x_coordinate < 395 and 0 <= y_coordinate < 500:
                node_in_pos = (x_coordinate, y_coordinate)
                if node_in_pos not in parent_path:
                    go_to = Node(node_in_pos[0], node_in_pos[1])
                    direction_to_go = way
                    go_to.parent_node = current_node
                    parent_path[(go_to.x_coord, go_to.y_coord)] = current_node
                    current_node_pixel, go_to_node_pixel = fetch_terrain_image_content(image_of_terrain, current_node,
                                                                                       go_to)
                    # print(current_node_data, go_to_node_data)
                    current_node_speed, go_to_node_speed = fetch_speed_content(current_node_pixel, go_to_node_pixel,
                                                                               season)
                    # print(current_node_speed, go_to_node_speed)
                    # print(len(elevation_file))
                    elevation = fetch_elevation_content(elevation_file, current_node, go_to)
                    valley, distance = dist_between_two_points(elevation, direction_to_go)
                    time_taken = calculate_time_taken(valley, distance, current_node_speed, go_to_node_speed,
                                                      current_node, go_to, cost_till_now, goal_node)
                    frontier.put((time_taken, (go_to.x_coord, go_to.y_coord)))
            else:
                continue


def read_path_to_follow(path_to_follow, imag, elevation_file, output_file_name, season, image_t):
    '''
    To read the path that is given as input
    :param path_to_follow: input image file information
    :param imag: image information
    :param elevation_file: elevation information
    :param output_file_name: png file name where the output has to be printed
    :param season: season
    :param image_t: image file
    :return: None
    '''
    with open(path_to_follow) as f:
        list_of_nodes = []
        coordinates = []
        for row in f:
            coordinates = row.split()
            coordinates = [int(x) for x in coordinates]
            # print(coordinates, type(coordinates))
            x = Node(coordinates[0], coordinates[1])
            list_of_nodes.append(x)
        start_node = list_of_nodes[0]
        # print(start_node.x_coord, start_node.y_coord)
        visit_nodes(list_of_nodes[1:], imag, elevation_file, output_file_name, season, image_t, start_node)


def draw_path(image_Input, path, node, output_file_name):
    '''
    To draw the path on the output image
    :param image_Input: input image
    :param path: path
    :param node: node to be drawn point on
    :param output_file_name: output image file name
    :return: None
    '''
    ip = image_Input.load()
    while True:
        ip[node[0], node[1]] = (255, 0, 0, 255)
        if path[(node[0], node[1])] is not None:
            node = path[(node[0], node[1])]
        else:
            break
    image_Input.save(output_file_name)


def draw_points(path_list, image_input, output_file_name):
    '''
    To highlight the nodes on the path
    :param path_list: path to be followed
    :param image_input: input image
    :param output_file_name: output file name
    :return: None
    '''
    ip = image_input.load()
    neighbours = [[-1, 0, 'left'], [-1, -1, 'top-left'], [-1, 1, 'bottom-left'],
                  [0, -1, 'top'], [0, 1, 'bottom'],
                  [1, 0, 'right'], [1, -1, 'top-right'], [1, 1, 'bottom-right']]
    for item in path_list:
        for neighbour in neighbours:
            ip[item.x_coord + neighbour[0], item.y_coord + neighbour[1]] = (0, 0, 98)
            ip[item.x_coord + 2 * neighbour[0], item.y_coord + 2 * neighbour[1]] = (0, 0, 98)

    image_input.save(output_file_name)


def visit_nodes(path_list, imag, elevation_file, output_file_name, season, image_of_terrain, start_node):
    '''
    To follow the input file and visit those nodes in the A* search
    :param path_list: path list
    :param imag: image information
    :param elevation_file: elevation file data
    :param output_file_name: output image file name
    :param season: season
    :param image_of_terrain: input image
    :param start_node: start node
    :return: None
    '''
    global total_time
    for node in path_list:
        cost, path = a_star_search(start_node, node, imag, elevation_file, season)
        total_time += cost
        draw_path(image_of_terrain, path, (node.x_coord, node.y_coord), output_file_name)
        draw_points(path_list, image_of_terrain, output_file_name)
        start_node = node


def bfs(pixelMap, season, go_here, elevation_file, i, j, pixel_point, depth, neighbours):
    '''
    Breadth First Search to color the pixels to convert in case of winter or spring
    :param pixelMap: image information
    :param season: season
    :param go_here: neighbours that can be visited
    :param elevation_file: elevation information
    :param i: row index
    :param j: column index
    :param pixel_point: start index to be checked from
    :param depth: depth of each pixel
    :param neighbours: neibouring pixel information that might be visited after calculation
    :return: image information
    '''
    queue = []
    queue.append((pixel_point, depth))
    maximum_depth = 0
    winter_rgb = (0, 0, 255, 255)
    if season == 'spring':
        maximum_depth = 15
    else:
        maximum_depth = 3
    while len(queue) > 0:
        current = queue.pop(0)
        pixel = current[0]
        pixel_depth = current[1]
        if pixel_depth < maximum_depth:
            for neighbour in neighbours:
                check_this_neighbour = (neighbour[0] + pixel[1],
                                        neighbour[1] + pixel[0])
                x_coordinate, y_coordinate = check_this_neighbour[0], check_this_neighbour[1]
                if 0 <= x_coordinate < 395 and 0 <= y_coordinate < 500:
                    if season == 'spring':
                        if pixelMap[x_coordinate, y_coordinate] in go_here:
                            if elevation_file[y_coordinate][x_coordinate] - elevation_file[i][j] <= 5.0:
                                pixelMap[x_coordinate, y_coordinate] = (114, 104, 30, 255)
                                queue.append(((y_coordinate, x_coordinate), pixel_depth+1))
                    elif season == 'winter':
                        if pixelMap[x_coordinate, y_coordinate] == winter_rgb:
                            pixelMap[x_coordinate, y_coordinate] = (170, 200, 255, 255)
                            queue.append(((y_coordinate, x_coordinate), pixel_depth+1))
        else:
            continue
    return pixelMap



def change_image(image, elevation_file, season):
    '''
    To edit the image in case of spring or winter
    :param image: input image
    :param elevation_file: elevation information
    :param season: season
    :return: None
    '''
    go_here = {(248, 148, 18, 255), (255, 192, 0, 255), (255, 255, 255, 255), (2, 208, 60, 255), (2, 136, 40, 255),
               (71, 51, 3, 255), (0, 0, 0, 255)}

    neighbours = [[-1, 0, 'left'], [-1, -1, 'top-left'], [-1, 1, 'bottom-left'],
                  [0, -1, 'top'], [0, 1, 'bottom'],
                  [1, 0, 'right'], [1, -1, 'top-right'], [1, 1, 'bottom-right']]

    pixelMap = image.load()

    img = Image.new(image.mode, image.size)
    pixelNew = img.load()

    pixel = None
    for i in range(image.size[1]):
        for j in range(image.size[0]):
            if pixelMap[j, i] == (0, 0, 255, 255):
                for neighbour in neighbours:
                    check_this_neighbour = (neighbour[0] + j,
                                            neighbour[1] + i)
                    x_coordinate, y_coordinate = check_this_neighbour[0], check_this_neighbour[1]
                    if 0 <= x_coordinate < 395 and 0 <= y_coordinate < 500:
                        if pixelMap[x_coordinate, y_coordinate] in go_here:
                            bfs(pixelMap, season, go_here, elevation_file, i, j, (i, j), 0, neighbours)
                    else:
                        continue

def read_inputs(arguments, start_time):
    '''
    To read all the inputs from the arguments
    :param arguments: system arguments list
    :param start_time: start time of the code
    :return: None
    '''
    image_of_terrain = arguments[1]
    elevation_file = arguments[2]
    path_to_follow = arguments[3]
    # print(path_to_follow)
    season = arguments[4]
    output_file_name = arguments[5]

    imag = read_image(image_of_terrain)
    image_t = Image.open(image_of_terrain)
    ele_list = read_elevation_txt(elevation_file)
    season = season.lower()
    if season == 'summer' or season == 'fall':
        read_path_to_follow(path_to_follow, imag, ele_list, output_file_name, season, image_t)
        # visit_nodes(path_list, imag, ele_list, output_file_name, season, image_of_terrain)
        print('Time taken for ', season, ' is: ', time.time() - start_time)
        print('Total distance for ', season, ' is: ', total_distance)
    elif season == 'winter' or season == 'spring':
        change_image(image_t, ele_list, season)
        image_t.save(output_file_name)
        imag = Image.open(image_of_terrain).load()
        image_t = Image.open(output_file_name)
        read_path_to_follow(path_to_follow, imag, ele_list, output_file_name, season, image_t)
        print('Time taken for the code is: ', time.time() - start_time)
        print('Total distance for', season, 'is:', total_distance/8, 'meters')


def main():
    '''
    Main method
    :return: None
    '''
    start_time = time.time()
    read_inputs(sys.argv, start_time)


if __name__ == '__main__':
    main()