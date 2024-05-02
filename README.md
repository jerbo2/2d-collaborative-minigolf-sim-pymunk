# 2d-collaborative-minigolf-sim-pymunk
A nonholonomic robot control sim in Pygame / Pymunk where a user and a robot take turns hitting the ball. The robot can path find using Dijkstra's algorithm and intuitively select its shots based on the position of its surroundings.

`pip install -r requirements.txt`
`python3 sim.py`

Obstacles can be configured by editing the obstacles array in sim.py line 205. Click and drag on the ball to apply force to the ball in a specified direction. After the ball stops after your shot, the robot should take over for one shot. The ball will reset to the starting position if it goes into the hole. 
