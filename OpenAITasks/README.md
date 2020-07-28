# Information
<h4>Name: Varun Lalwani<br>
Email: vlalwani31@gmail.com<br>
Source: Open AI Gym<br></h4>

# Tasks
### Cart Pole
- This task involes balancing a pole upwards by moving a cart in left or right direction
- The task ends if the pole falles below an angle on 15 degress or the cart moves 2.4 m in either left or right direction
- The task is solved using Deep Q-Learning

### Mountain Car
- This task involves moving a car to the top of the mountain by moving it in left or right direction
- The task ends if the car reaches the top of the mountain
- The task ends if the car is unable to reach the top of the mountain after 1000 steps
- The task is solved using Q-Learning

### Pendulum
- This task involves balancing a Pendulum in a upright position for a period of time
- The task ends when the task is accomplished or when the time frame to complete the task is reached
- The task has continuous observation space and continuous action space
- Standard Q-learning and DQN methods are incompetent to perform such tasks
- The observation space is 3-Dimensional and action space is 1-Dimensional.
- We use Deep Deterministic Policy gradient(DDPG) to solve this task

### Bi-Pedal Walker(Normal version)
- This task involves teaching a two legged robot walk on a plain field
- The task ends when the robot reaches the goal
- The task has continuous observation space and continuous action space
- Standard Q-learning and DQN methods are incompetent to perform such tasks
- The observation space is 24-Dimensional and action space is 4-Dimensional.
- We use Deep Deterministic Policy gradient(DDPG) to solve this task
