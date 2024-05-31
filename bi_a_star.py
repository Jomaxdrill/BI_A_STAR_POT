#* WE ARE ASSUMING UNITS IN CM FOR CONVENIENCE IN GRID COMPUTATION
import numpy as np
import heapq as hq
import time
from maze import distance,rotation_vectors_by, check_in_obstacle
from plotting import plot_paths
STEP_SIZE = 10
DIV_ANGLE = 12
TH_ANGLE = 360 // DIV_ANGLE #*must be between 0-45 degrees
NORM_SET = [-2, 1, 0, 1, 2]
ACTION_SET = TH_ANGLE * np.array(NORM_SET)
ANGLE_VALUES_RAW = list(range(0,  DIV_ANGLE // 2  + 1)) + list(range(-(DIV_ANGLE // 2  -1), 0))
ANGLE_VALUES = TH_ANGLE * np.array(ANGLE_VALUES_RAW)
ACTION_INDEX = list(range(0, len(ACTION_SET)))
MAX_ITER = 200000 # max operations to perform
NORM_VECTOR = (1, 0) #vector for orientation of the robot's front-facing
ROT_MATRICES = np.array([ rotation_vectors_by(angle) for angle in ANGLE_VALUES ])
MAX_ITERATIONS = 150000
#GENERAL VARIABLES FOR BI A*
full_path = []
bi_trees = { 'A': {
	"open_list": [],
	"close_list": [],
	"close_tracking": set(),
	"open_tracking": {},
	"goal_path":[],
	"counter": 0
	} ,
	'B': {
	"open_list": [],
	"close_list": [],
	"close_tracking": set(),
	"open_tracking": {},
	"goal_path":[],
	"counter": 0
	}
}
hq.heapify(bi_trees["A"]["open_list"])
hq.heapify(bi_trees["B"]["open_list"])

def round_float(number):
	"""
	This function rounds a float number to the nearest integer.

	Args:
		number (float): The float number to round.

	Returns:
		int: The nearest integer to the input float number.

	"""
	if number % 1 < 0.5:
		return int(number)
	return int(number) + 1

def action_move(current_node, action, current_tree, target, constraints, option, matrix):
	"""
	Args:
		current_node (Node): Node to move

	Returns:
		Node: new Node with new configuration and state
	"""
	BORDER, WIDTH, HEIGHT = constraints
	state_moved = apply_action(current_node[5:], action)
	#*check when actions could inmediately be out of bounds
	if (state_moved[0] <= 0 or state_moved[1] <= 0) or (state_moved[0] >= WIDTH or state_moved[1] >= HEIGHT):
		return None
	# *check new node is in obstacle space
	if check_in_obstacle(state_moved[0:2], option, constraints, matrix):
		return None
	# *check by the state duplicate values between the children
	node_already_visited = state_moved[0:2] in bi_trees[current_tree]['close_tracking']
	if node_already_visited:
		return None
	#create new node
	new_cost_to_come = current_node[1] + STEP_SIZE
	new_cost_to_go = distance(state_moved[0:2], target) #heuristic function
	new_total_cost =  new_cost_to_come + new_cost_to_go
	new_node = (new_total_cost, new_cost_to_come, new_cost_to_go) + (-1, current_node[3]) + state_moved
	return new_node

def apply_action(state, type_action):
	"""
	Applies the given action to the given state.

	Args:
		state (tuple): The current state of the robot, as a tuple of its x and y coordinates and its orientation.
		type_action (str): The type of action to apply, as a string.

	Returns:
		tuple: The new state of the robot, as a tuple of its x and y coordinates and its orientation.

	"""
	x_pos, y_pos, theta = state
	#check action is valid
	action_to_do = np.where(ANGLE_VALUES == type_action)[0][0]
	if action_to_do is None:
		return None
	#get the proper orientation of the robot, its current front
	rotation_index = np.where(ANGLE_VALUES == theta)[0][0]
	vector_front = np.dot(ROT_MATRICES[rotation_index], NORM_VECTOR)
	new_vector = np.dot(ROT_MATRICES[action_to_do], vector_front) * STEP_SIZE
	#calculate new positions and orientation
	x_pos_new = round_float(x_pos + new_vector[0])
	y_pos_new = round_float(y_pos + new_vector[1])
	angle_degrees = theta + type_action
	if angle_degrees > 180:
		angle_degrees = angle_degrees - 360
	if angle_degrees <= -180:
		angle_degrees = angle_degrees + 360
	return (x_pos_new, y_pos_new, angle_degrees)
def trees_meet(node, current_tree):
	"""
	Checks if a given node is present in the open or close list of the other tree.

	Parameters:
	node (tuple): The node to check for in the other tree. It should be a tuple representing the x and y coordinates of the node.
	current_tree (str): The current tree being evaluated. It should be a string representing either 'A' or 'B'.

	Returns:
	list: A list containing two boolean values. The first value indicates if the node if the generated node is in the other tree.
	The second value indicates if the node is present in the open list of the other tree.
	"""
	check_tree = 'A' if current_tree == 'B' else 'B'
	if node in bi_trees[check_tree]["close_tracking"]:
		return [True, True]
	if node in bi_trees[check_tree]["open_tracking"]:
		return [True, False]
	return [False, False]
#*NODE STRUCTURE----------------------------------------------------------------
# ?(cost_total, cost_to_come,cost_to_go, index, parent, x_pos, y_pos, angle)
def bi_a_star(initial_state, goal_state, option, constraints, matrix):
	"""
	Performs the Bi-directional A* algorithm to find a path from the initial state to the goal state.

	Args:
		initial_state (tuple): The initial state of the robot, as a tuple of its x and y coordinates and its orientation.
		goal_state (tuple): The goal state of the robot, as a tuple of its x and y coordinates and its orientation.
		option (str): The option for the algorithm, as a string.
		constraints (tuple): The constraints for the algorithm, as a tuple.
		matrix (numpy.ndarray): The matrix representing the obstacles in the environment.

	Returns:
		dict: A dictionary containing the node that joins the trees, the time taken to find the solution, and other relevant information.
	"""
	print("Bi-A* start!!\n")
	start_time = time.time()
	iterations = 0
	distance_init_goal = distance(initial_state, goal_state)
	cost_init = (distance_init_goal, 0, distance_init_goal)
	initial_node_A = cost_init + (0, None) + initial_state
	initial_node_B = cost_init + (0, None) + goal_state
	# Add initial node to the heaps
	hq.heappush(bi_trees["A"]["open_list"], initial_node_A)
	hq.heappush(bi_trees["B"]["open_list"], initial_node_B)
	bi_trees["A"]["open_tracking"][initial_node_A[5:7]] = (0,0)
	bi_trees["B"]["open_tracking"][initial_node_B[5:7]] = (0,0)
	current_tree = 'A'
	print("node counting:\n")
	while iterations < MAX_ITERATIONS:
		if not len(bi_trees[current_tree]["open_list"]):
			break
		print(f' iteration: {iterations},tree: {current_tree},counter: {bi_trees[current_tree]["counter"]} ', end='\r')
		current_node = bi_trees[current_tree]["open_list"][0]
		hq.heappop(bi_trees[current_tree]["open_list"])
		# Mark node as visited
		bi_trees[current_tree]["close_list"].append(current_node)
		bi_trees[current_tree]["close_tracking"].add(current_node[5:7])
		target = goal_state if 'A' else initial_state
		for action in ACTION_SET:
			child = action_move(current_node, action, current_tree, target, constraints, option, matrix)
			if not child:
				continue
			#*check if trees meet
			trees_converge = trees_meet(child[5:7], current_tree)
			if trees_converge[0]:
				#*if node was found in the open list of the other tree, move that node to the closedlist of the other tree
				if not trees_converge[1]:
					check_tree = 'A' if current_tree == 'B' else 'B'
					#*find the node in the open list of the other tree
					idx = 0
					for node in bi_trees[check_tree]["open_list"]:
						if node[5:7] == child[5:7]:
							break
						idx += 1
					node_to_move = bi_trees[check_tree]["open_list"][idx]
					bi_trees[check_tree]["close_list"].append(node_to_move)
					bi_trees[check_tree]["close_tracking"].add(node_to_move[5:7])
				#*add inmediately this node to the current tree it was generated
				child_join = child[0:3] + (bi_trees[current_tree]['counter'],) + child[4:]
				bi_trees[current_tree]['counter'] += 1
				bi_trees[current_tree]["close_list"].append(child_join)
				bi_trees[current_tree]["close_tracking"].add(child_join[5:7])
				end_time = time.time()
				time_sol = end_time-start_time
				print(f'A node that join the tree has been found it is {child_join}')
				print(f'DONE in {time_sol} seconds.\n')
				return { 'node_join': child_join,
						'time': time_sol
				}
			#* Check if child is in open list generated nodes
			in_open_list = bi_trees[current_tree]['open_tracking'].get(child[5:7], None)
			if not in_open_list:
				bi_trees[current_tree]['counter'] += 1
				child_to_enter = child[0:3] + (bi_trees[current_tree]['counter'],) + child[4:]
				hq.heappush(bi_trees[current_tree]["open_list"], child_to_enter)
				bi_trees[current_tree]["open_tracking"][child[5:7]] = (child[1], bi_trees[current_tree]['counter'])
			#* check if cost to come is greater in node in open list
			elif in_open_list[0] > child[1]:
				#?create a new node with cost 0 so it can be visited quickly and current node will never be visited
				child_to_enter = (0,0,0) + (in_open_list[1],) + child[4:]
				hq.heappush(bi_trees[current_tree]["open_list"], child_to_enter)
		#* swap trees
		current_tree = 'A' if current_tree == 'B' else 'B'
		iterations += 1
	end_time = time.time()
	print(f'No solution found. Process took {end_time - start_time} seconds.')
	return { 'time': end_time - start_time }

def backtracking(node_check, tree):
	"""Generate the path from the initial node to the goal state.

	Args:
		node (Node): Current node to evaluate its parent (previous move done).
	Returns:
		Boolean: True if no more of the path are available
	"""
	#find the connector node in the tree
	node_start = None
	for node_find in bi_trees[tree]["close_list"]:
		if node_check[5:7] == node_find[5:7]:
			node_start = node_find
			break
	while node_start is not None:
		bi_trees[tree]["goal_path"].append(node_start[5:])
		parent_at = 0
		for node_check in bi_trees[tree]["close_list"]:
			if node_check[3] == node_start[4]:
				break
			parent_at += 1
		node_start = bi_trees[tree]["close_list"][parent_at] if parent_at < len(bi_trees[tree]["close_list"]) else None
	return True

def get_path_by(node_join):
	"""
	Generate the full path from the initial state to the goal state by combining the paths from both trees.

	Parameters:
	node_join (tuple): The node that connects the two trees. It should be a tuple representing the x and y coordinates of the node.

	Returns:
	list: A list of tuples representing the coordinates of the nodes in the full path. The path is generated by combining the paths from both trees.
	"""
	#find the path for each tree beginning with the node that joins them
	backtracking(node_join, 'A')
	backtracking(node_join, 'B')
	full_path = bi_trees["A"]['goal_path'][::-1] + bi_trees["B"]['goal_path'][1:]
	return full_path
def plan_bi_a_star(init, goal, option, constraints, obstacles, matrix=None):
	"""
	This function plans a path using the Bidirectional A* algorithm.

	Parameters:
	init (tuple): The initial state of the robot, as a tuple of its x and y coordinates and its orientation.
	goal (tuple): The goal state of the robot, as a tuple of its x and y coordinates and its orientation.
	option (str): The option for the algorithm, as a string.
	constraints (tuple): The constraints for the algorithm, as a tuple.
	obstacles (list): A list of tuples representing the coordinates of the obstacles in the environment.
	matrix (numpy.ndarray, optional): The matrix representing the obstacles in the environment.

	Returns:
	dict: A dictionary containing the solution path, time taken, step size, algorithm name, and color.
	"""
	initial_hit = check_in_obstacle(init[0:2], option, constraints, matrix)
	goal_hit = check_in_obstacle(goal[0:2], option, constraints, matrix)
	#verify validity of positions
	hit = initial_hit or goal_hit
	if hit:
		print("Start or goal are already in obstacle space. Please run the program again.")
		return None
	result = bi_a_star(init, goal, option, constraints, matrix)
	if result.get('node_join', None) is not None:
		full_path = get_path_by(result['node_join'])
	else:
		sol_A = list(bi_trees['A']['close_tracking'])
		sol_B = list(bi_trees['B']['close_tracking'])
		full_path = [(x, y, 0) for x, y in sol_A] + [(x, y, 0) for x, y in sol_B]
	total_sol = { 'sol_path': full_path,
				'time': result['time'],
				'step': STEP_SIZE,
				'name': 'Bidirectional A*',
				'color' : 'g'}
	sol_graphs = {'bi_a_star': total_sol}
	#plot_paths(sol_graphs, obstacles, constraints)
	return sol_graphs