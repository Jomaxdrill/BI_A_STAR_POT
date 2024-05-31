from pot_field import plan_potential_field
from bi_a_star import plan_bi_a_star
from plotting import plot_paths
"""SET THE MAIN VARIABLES """
#*Coordinates in format x,y and every distance unit here are in CM!!
INITIAL= (0, 150,0)
GOAL = (600, 150, 180) #if using bi A* remember set angle in the proper direction to let the tree spread
RADIUS_ROBOT = 22
border = 5
width = 600
height = 300
OPTION = 3 #!dont move value
border_total = border  + RADIUS_ROBOT
OBSTACLES = ((112,242.5,40), (263,90,70), (445,220,37.5)) #FOR POTENTIAL FIELD ALGORITHM AND PLOTTING
#*Apply the algorithm and get graphs for each
result_1 = plan_bi_a_star(INITIAL, GOAL, OPTION,(border_total, width, height),OBSTACLES)
result_2 = plan_potential_field(INITIAL, GOAL, OBSTACLES, OPTION, (border_total, width, height))
results_total = {**result_1, **result_2}
plot_paths(results_total, OBSTACLES, (border_total, width, height))