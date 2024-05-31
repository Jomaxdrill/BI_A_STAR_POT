
#* WE ARE ASSUMING UNITS IN CM FOR CONVENIENCE IN GRID COMPUTATION
import numpy as np
import time
from maze import distance_exact, distance,rotation_vectors_by,check_in_obstacle
from plotting import plot_paths
K_attract = 50 # attraction force parameter of goal
K_obstacle = 50000000 #repulsion force parameter
RAD_GOAL = 5
STEP_SIZE = 5
#*For action set
DIV_ANGLE = 12
TH_ANGLE = 360 // DIV_ANGLE #*must be between 0-45 degrees
NORM_SET = [-2, 1, 0, 1, 2]
ACTION_SET = TH_ANGLE * np.array(NORM_SET)
ANGLE_VALUES_RAW = list(range(0,  DIV_ANGLE // 2  + 1)) + list(range(-(DIV_ANGLE // 2  -1), 0))
ANGLE_VALUES = TH_ANGLE * np.array(ANGLE_VALUES_RAW)
ACTION_INDEX = list(range(0, len(ACTION_SET)))
MAX_ITER = 10000 # max operations to perform
NORM_VECTOR = (1, 0) #vector for orientation of the robot's front-facing
ROT_MATRICES = np.array([ rotation_vectors_by(angle) for angle in ANGLE_VALUES ])

def attractive_force(goal,current_position):
	"""
	Calculates the attractive force towards the goal.

	Parameters:
	goal (tuple): A tuple representing the goal's position in the format (x, y).
	current_position (tuple): A tuple representing the current position of the agent in the format (x, y).

	Returns:
	float: The attractive force towards the goal.

	The attractive force is calculated using the formula: 0.5 * K_attract * (distance_exact(goal, current_position)).
	"""
	# Calculate attractive force towards the goal
	return 0.5* K_attract* (distance_exact(goal, current_position))
def repulsive_force(obstacles, current_position, contour):
	"""
	Calculates the repulsive force from obstacles.

	Parameters:
	obstacles (list): A list of tuples, where each tuple represents an obstacle.
					Each tuple contains the x and y coordinates of the obstacle's center,
					and the radius of the obstacle.
					For example: [(x1, y1, radius1), (x2, y2, radius2),...]

	current_position (tuple): A tuple representing the current position of the agent in the format (x, y).

	contour (float): The distance beyond which the obstacle's repulsive force becomes zero.

	Returns:
	float: The total repulsive force from all obstacles.

	The repulsive force is calculated using the formula:
	0.5 * K_obstacle * ((1/distance_to_obstacle_center) - 1/(border_repulsion))**2
	where distance_to_obstacle_center is the distance from the current position to the obstacle's center,
	and border_repulsion is the distance beyond which the obstacle's repulsive force becomes zero.
	"""
	# Calculate repulsive force from obstacles
	repulsive_force = 0
	for obs_elem in obstacles:
		border_repulsion = obs_elem[2] + contour
		distance_to_obstacle_center = distance_exact(obs_elem[0:2], current_position)
		#assumption: a bigger obstacle will have more repulsion force, use radius as parameter
		#0 if distance_to_obstacle_center >= border_repulsion else
		repulsive_partial = 0 if distance_to_obstacle_center >= border_repulsion else 0.5*  (K_obstacle) * ((1/distance_to_obstacle_center) - 1/(border_repulsion))**2
		repulsive_force += repulsive_partial
	return repulsive_force


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
	x_pos_new = round(x_pos + new_vector[0])
	y_pos_new = round(y_pos + new_vector[1])
	angle_degrees = theta + type_action
	if angle_degrees > 180:
		angle_degrees = angle_degrees - 360
	if angle_degrees <= -180:
		angle_degrees = angle_degrees + 360
	return (x_pos_new, y_pos_new, angle_degrees)

def potential_field(initial, goal, obstacles ,constraints):
	"""
	Implements the potential field algorithm for path planning.

	Parameters:
	initial (tuple): The initial position of the robot in the format (x, y, theta).
	goal (tuple): The goal position of the robot in the format (x, y).
	obstacles (list): A list of tuples, where each tuple represents an obstacle.
					Each tuple contains the x and y coordinates of the obstacle's center,
					and the radius of the obstacle.
					For example: [(x1, y1, radius1), (x2, y2, radius2),...]
	constraints (tuple): The constraints of the environment in the format (border, width, height).

	Returns:
	dict: A dictionary containing the solution path, time taken, step size, name, and color.

	The function uses the potential field algorithm to find the optimal path from the initial position to the goal,
	avoiding obstacles. It calculates the attractive force towards the goal and the repulsive force from obstacles,
	and moves the robot to the position where the total force is minimized.
	"""
	BORDER, WIDTH, HEIGHT = constraints
	goal_reached = False
	sol_path = []
	start_time = time.time()
	counter_nodes = 0
	less_cost_node = None
	current_pos = initial
	sol_path.append(current_pos)
	print('Potential field start!!')
	while not goal_reached and counter_nodes < MAX_ITER:
		dist_to_goal = distance(goal[0:2], current_pos)
		print(f"Distance to goal:{dist_to_goal} iteration:{counter_nodes}", end="\r")
		goal_reached = dist_to_goal <= RAD_GOAL
		if goal_reached:
			end_time = time.time()
			time_sol = end_time-start_time
			print(f'DONE in {time_sol} seconds.\n')
			return { 'sol_path': sol_path,
						'time': end_time-start_time,
					}
		cost_move = np.inf
		less_cost_node = None
		#* move to where the gradient is lower between this set of moves
		for action in ACTION_SET:
			state_moved = apply_action(current_pos, action)
			#*check when actions could inmediately be out of bounds,
			if (state_moved[0] < 0 or state_moved[1] < 0):
				continue
			if (state_moved[0] >= WIDTH or state_moved[1] >= HEIGHT-BORDER or state_moved[1] <= BORDER):
				continue
			current_attraction = attractive_force(goal, state_moved)
			current_repulsion = repulsive_force(obstacles, state_moved, BORDER)
			total_force = current_attraction + current_repulsion
			if total_force < cost_move:
				cost_move = total_force
				less_cost_node = state_moved
		if less_cost_node is not None:
			sol_path.append(less_cost_node)
			current_pos = less_cost_node
		counter_nodes += 1
	end_time = time.time()
	print(f'No optimal solution found. Process took {end_time - start_time} seconds.')
	return { 'sol_path': sol_path,
				'time': end_time-start_time}


def plan_potential_field(init, goal, obstacles,option,constraints, matrix=None):
	"""
	Plans a path using the Potential Field algorithm.

	Parameters:
	init (tuple): The initial position of the robot in the format (x, y, theta).
	goal (tuple): The goal position of the robot in the format (x, y).
	obstacles (list): A list of tuples, where each tuple represents an obstacle.
					Each tuple contains the x and y coordinates of the obstacle's center,
					and the radius of the obstacle.
					For example: [(x1, y1, radius1), (x2, y2, radius2),...]
	option (str): The option for the obstacle representation.
	constraints (tuple): The constraints of the environment in the format (border, width, height).
	matrix (numpy.ndarray, optional): A 2D matrix representing the environment. Defaults to None.

	Returns:
	dict: A dictionary containing the solution path, time taken, step size, name, and color.

	The function first checks if the initial or goal positions are already in an obstacle space.
	If not, it calls the potential_field function to find the optimal path.
	The resulting path, time taken, step size, name, and color are stored in a dictionary and returned.
	"""
	initial_hit = check_in_obstacle(init[0:2], option, constraints, matrix)
	goal_hit = check_in_obstacle(goal[0:2], option, constraints, matrix)
	#verify validity of positions
	hit = initial_hit or goal_hit
	if hit:
		print("Start or goal are already in obstacle space. Please run the program again.")
		return None
	result = potential_field(init, goal, obstacles ,constraints)
	total_sol = { 'sol_path': result['sol_path'],
				'time': result['time'],
				'step': STEP_SIZE,
				'name': 'Potential Field',
				'color' : 'r'}
	sol_graphs = {'potential': total_sol}
	#plot_paths(sol_graphs, obstacles, constraints)
	return sol_graphs

