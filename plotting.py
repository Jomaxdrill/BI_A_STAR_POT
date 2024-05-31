import matplotlib.pyplot as plt
import numpy as np
def plot_paths(paths_sol, obstacles, constraints):
	"""
	This function plots multiple paths on a 2D graph for visualization.

	Parameters:
	paths_sol (list): A list of tuples, where each tuple represents a path. Each path is a list of (x, y) coordinates.

	Returns:
	None

	The function extracts the x and y coordinates from each path and plots them on a graph using matplotlib.
	The paths are labeled with their respective names ('Initial Path', 'Optimal Path', 'Smoothed Path') for clarity.
	The graph is then configured with a title, axis labels, legend, and grid for better visualization.
	Finally, the graph is displayed using plt.show().
	"""
	_, WIDTH, HEIGHT = constraints
	_, axes = plt.subplots()
	axes.set(xlim=(0, WIDTH), ylim=(0, HEIGHT))
	for obs_elem in obstacles:
		x_obs, y_obs, r_obs = obs_elem
		draw_obs = plt.Circle(( x_obs, y_obs ), r_obs )
		axes.add_artist( draw_obs )
	name_total = []
	for index_key in paths_sol:
		coords = paths_sol[index_key]['sol_path']
		color = paths_sol[index_key]['color']
		time = paths_sol[index_key]['time']
		step = paths_sol[index_key]['step']
		name = paths_sol[index_key]['name']
		name_total.append(name)
		x_elements = [x for x, _,_ in coords]
		y_elements = [y for _, y,_ in coords]
		#?Print vectors on graph
		# u_diff = np.diff(x_elements)
		# v_diff = np.diff(y_elements)
		# pos_x = x_elements[:-1] + u_diff/2
		# pos_y = y_elements[:-1] + v_diff/2
		# norm = np.sqrt(u_diff**2 + v_diff**2)
		#axes.quiver(pos_x, pos_y, u_diff/norm, v_diff/norm, angles="xy", zorder=1, pivot="mid")
		axes.plot(x_elements, y_elements,f'{color}.', label = f'{name}, t: {time} [s], step: {step} [cm]')
	plt.title(f"{' vs '.join(name_total)} solution path")
	plt.xlabel("X position [cm]")
	plt.ylabel("Y position [cm]")
	plt.legend()
	plt.grid()
	plt.show()
