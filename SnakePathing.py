import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
import sys
from heapq import *
import networkx as nx

pygame.init()

if len(sys.argv) <= 1:
    num_obstacles = 15
    mode = 'astar'
else:
    mode = sys.argv[1]
    num_obstacles = sys.argv[2]
    

# Defining the colors that the snake and food will use.  
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
 
# Width of the game board (in tiles). 
WIDTH  = 20
# Height of the game board (in tiles).
HEIGHT = 20

# Size of each tile (in pixels).
STEPSIZE = 20

# How fast the game runs. Higher values are faster. 
CLOCK_SPEED = 100
 
# Making a pygame display. 
dis = pygame.display.set_mode((WIDTH*STEPSIZE,HEIGHT*STEPSIZE))
pygame.display.set_caption('Snake!')

# Initial variables to store the starting x and y position,
# and whether the game has ended. 
game_over = False 
x1 = 5
y1 = 5
snake_list = [(x1,y1)]
snake_len = 1 
x1_change = old_x1_change = 0       
y1_change = old_y1_change = 0

# PyGame clock object.  
clock = pygame.time.Clock()

food_eaten = True

# Random obstacles, if desired. 
obstacles = [(np.random.randint(low=0, high=WIDTH),np.random.randint(low=0, high=HEIGHT)) for i in range(int(num_obstacles))]

# This method is a wrapper for the various AI methods. 
# right now it just moves the snake randomly regardless
# of the board state, because none of those methods are 
# filled in yet. 
# Bstate is a matrix representing the game board:
### Array cells with a 0 are empty locations. 
### Array cells with a -1 are the body of the snake.
### The cell marked with a -2 is the head of the snake.
### The cell marked with a 1 is the food.
def get_AI_moves(ai_mode, bstate):
    if ai_mode == 'rand':
        return random_AI(bstate)
    elif ai_mode == 'greedy':
        return greedy_AI(bstate)
    elif ai_mode == 'astar':
        return astar_AI(bstate)  
    elif ai_mode == 'dijkstra':
        return dijkstra_AI(bstate)  
    elif ai_mode == 'backt':
        return backt_AI(bstate)    
    else:
        raise NotImplementedError("Not a valid AI mode!\nValid modes are rand, greedy, astar, dijkstra, and backt.")    


# These are the methods you will fill in. 
# Each method takes in a game board (as described above), and
# should output a series of moves. Valid moves are: 
# (0,1),(0,-1),(1,0), and (-1,0). This means if you want to
# move in any more complicated way, you need to convert the move
# you want to make into a sequence like this one.
# For example, if I wanted my snake to move +5 in the x direction and +3
# in the y direction, I could return 
# [(0,1),(0,1),(0,1),(0,1),(0,1),(1,0),(1,0),(1,0)].

# Several of these methods demonstrate how to get the source
# and target locations, but currently do not use this information. 

def Manhattan_distance(source, target):
    # Calculate Manhattan distance (heuristic)
    return abs(source[0] - target[0]) + abs(source[1] - target[1])

def astar_AI(bstate):
    # Find the position of the snake's head (where bstate == -2)
    source = tuple(np.array(np.where(bstate == -2)).T[0])
    # Find the position of the food (where bstate == 1)
    target = tuple(np.array(np.where(bstate == 1)).T[0])

    #Priority Queue to explore nodes, starting with the source node
    priority_q = []
    heappush(priority_q, (0, source, []))

    #keep track of visited nodes
    visited = set()
    # Dictionary to track the lowest g_cost for each node
    g_costs = {source: 0}

    while priority_q:
        current_cost, current_pos, path = heappop(priority_q)

        # If snake gets to the target, return the path taken
        if current_pos == target:
            return path
        
        if current_pos in visited:
            continue

        visited.add(current_pos)

        # Explore neighbors (up, down, left, right) with wraparound
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = ((current_pos[0] + dx) % bstate.shape[0], (current_pos[1] + dy) % bstate.shape[1])

            # Avoids obstacles and snake body
            if bstate[neighbor[0], neighbor[1]] != -1:
                # Calculate new g_cost for this neighbor
                new_g_cost = g_costs[current_pos] + 1  # Each move has a cost of 1
                if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                    g_costs[neighbor] = new_g_cost
                    m_cost = Manhattan_distance(neighbor, target)
                    total_cost = new_g_cost + m_cost
                    new_path = path + [(dx, dy)]
                    heappush(priority_q, (total_cost, neighbor, new_path))
    
    # Testing heuristic cost for astar to see if algorithm is correctly prioritizing nodes
    # print(f"Current Position: {current_pos}, Cost: {current_cost}, Path: {path}")
    # print(f"Neighbor: {neighbor}, g_cost: {new_g_cost}, h_cost: {m_cost}, Total Cost: {total_cost}")

    # If no path is found, return a random move
    return random_AI(bstate)

def dijkstra_AI(bstate):
    #Find the position of the snake's head (bstate == -2)
    source = tuple(np.array(np.where(bstate == -2)).T[0])
    #Find the position of the food (bstate == 1)
    target = tuple(np.array(np.where(bstate == 1)).T[0])

    #Create a graph from the current board state using NetworkX
    graph = nx.Graph()

    #Add nodes and edges for each open space (bstate == 0)
    for x in range(bstate.shape[0]):
        for y in range(bstate.shape[1]):
            #Ensure snake head avoids obstacles & snake body
            if bstate[x,y] != -1:
                #Add the current posistion as a node
                graph.add_node((x,y))

                # Add edges to neighboring positions (up, down, left, right)
                
                # Regular movement within the grid
                if bstate[(x - 1) % bstate.shape[0], y] != -1:  # Up (with wraparound)
                    graph.add_edge((x, y), ((x - 1) % bstate.shape[0], y))
                if bstate[(x + 1) % bstate.shape[0], y] != -1:  # Down (with wraparound)
                    graph.add_edge((x, y), ((x + 1) % bstate.shape[0], y))
                if bstate[x, (y - 1) % bstate.shape[1]] != -1:  # Left (with wraparound)
                    graph.add_edge((x, y), (x, (y - 1) % bstate.shape[1]))
                if bstate[x, (y + 1) % bstate.shape[1]] != -1:  # Right (with wraparound)
                    graph.add_edge((x, y), (x, (y + 1) % bstate.shape[1]))

    try:
        #Use Dijkstras alg to find the shortest path
        path = nx.dijkstra_path(graph, source, target)

        #Convert the path into a list of moves
        moves = []
        for i in range(1, len(path)):
            dx = path [i][0] - path[i - 1][0]
            dy = path [i][1] - path[i - 1][1]
            moves.append((dx, dy))
        return moves

    except nx.NetworkXNoPath:
        #If there is no path to food return a random move for now
        return random_AI(bstate)
            
def greedy_AI(bstate):
    #Find the position of the snake's head (bstate == -2)
    source = np.array(np.where(bstate == -2)).T[0]
    #Find the position of the food (bstate == 1)
    target = np.array(np.where(bstate == 1)).T[0]

    #Create an empty list to store the path of moves
    path = []

    #Loop until the snake reaches the food
    while (source != target).any():
        #calculate the x and y differences
        dx = target[0] - source[0]
        dy = target[1] - source[1]

        #Move in the x direction if the x coords !=
        if dx != 0:
            if dx > 0:
                #Move to the right
                path.append((1,0))
                source[0] += 1
            else:
                #Move to the left
                path.append((-1, 0))
                source[0] -= 1
        
        #Otherwise move in the y-direction if the y coords are !=
        elif dy != 0:
            if dy > 0:
                #Move downwards
                path.append((0, 1))
                source[1] += 1
            
            else:
                #Move upwards
                path.append((0, -1))
                source[-1] -= 1
    return path
    #return random_AI(bstate)
    
def random_AI(bstate):
    return [[(0,1),(0,-1),(1,0),(-1,0)][np.random.randint(low=0,high=4)]]

def backt_AI(bstate):
    source = np.array(np.where(bstate == -2))
    target = np.array(np.where(bstate == 1))
    return random_AI(bstate)
    

AI_moves = []

# Don't modify any code below this point!
# This code is not meant to be readable or understandable to you - it's the game
# engine and the particulars of moving the snake according to your AI.
# The particulars of the code below shouldn't matter to your AI code above.
# If you have questions, or if your AI code needs to be able to use any of the below,
# talk to Blake.  

while not game_over:


    if food_eaten:   
        fx = np.random.randint(low=0,high=WIDTH)
        fy = np.random.randint(low=0,high=HEIGHT)
        while (fx,fy) in snake_list or (fx,fy) in obstacles:
            fx = np.random.randint(low=0,high=WIDTH)
            fy = np.random.randint(low=0,high=HEIGHT)
        food_eaten = False
        
    dis.fill(white)
    
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if mode == 'human':    
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x1_change = -1
                    y1_change = 0
                elif event.key == pygame.K_RIGHT:
                    x1_change = 1
                    y1_change = 0
                elif event.key == pygame.K_UP:
                    y1_change = -1
                    x1_change = 0
                elif event.key == pygame.K_DOWN:
                    y1_change = 1
                    x1_change = 0
    if mode != 'human':
        if len(AI_moves) == 0:
            bstate = np.zeros((WIDTH,HEIGHT))
            for xx,yy in snake_list:
                bstate[xx,yy] = -1
            for xx,yy in obstacles:
                bstate[xx,yy] = -1
            bstate[snake_list[-1][0], snake_list[-1][1]] = -2
            bstate[fx,fy] = 1    
            AI_moves = get_AI_moves(mode, bstate)     
        x1_change, y1_change = AI_moves.pop(0)               
    if len(snake_list) > 1 :
        if ((snake_list[-1][0] + x1_change) % WIDTH) == snake_list[-2][0] and ((snake_list[-1][1] + y1_change)% HEIGHT) == snake_list[-2][1]:
            x1_change = old_x1_change
            y1_change = old_y1_change
    x1 += x1_change
    y1 += y1_change          
    
    x1 = x1 % WIDTH
    y1 = y1 % HEIGHT
    
    if x1 == fx and y1 == fy:
        snake_len += 1
        food_eaten = True
    
    snake_list.append((x1,y1))
    snake_list = snake_list[-snake_len:]
    
    if len(list(set(snake_list))) < len(snake_list) or len(set(snake_list).intersection(set(obstacles))) > 0:
        print("You lose! Score: %d" % snake_len)
        game_over = True
    else:
        sncols = np.linspace(.5,1.0, len(snake_list))
        for jj, (xx, yy) in enumerate(snake_list):
            pygame.draw.rect(dis, (0, 255*sncols[jj], 32*sncols[jj]), [xx*STEPSIZE, yy*STEPSIZE, STEPSIZE, STEPSIZE])

        for (xx, yy) in np.cumsum(np.array([[.5,.5],snake_list[-1]] + AI_moves), axis=0)[2:]:
            pygame.draw.circle(dis, red, (xx*STEPSIZE,yy*STEPSIZE), STEPSIZE/4)            
        
        if not food_eaten:
            pygame.draw.rect(dis, red, [fx*STEPSIZE, fy*STEPSIZE, STEPSIZE, STEPSIZE])
        
        for xx, yy in obstacles:
            pygame.draw.rect(dis, blue, [xx*STEPSIZE, yy*STEPSIZE, STEPSIZE, STEPSIZE])
        pygame.display.update()
     
        clock.tick(CLOCK_SPEED)
        
        old_x1_change = x1_change
        old_y1_change = y1_change
 
pygame.quit()
quit()