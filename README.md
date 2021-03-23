# Deep Flappy Bird

### Contributors
> Name: Fu-Lin Hsu     
> Name: James Hu    
> Name: Jonathan Choi


### Requirement
- Python 3.x
- Pygame
- Pytorch
- Numpy

### Environment

#### env.py
The Flappy Bird Game Environment    
To load resources (images) for the game, specify the __BASE_PATH__ parameter of __load()__  
Use __frame_step(input_actions)__ to interact with the game and get the next observation    
The game ends when the return value, __terminal__, of frame_step is True

#### env_utils.py
The utilities for establishing the environment

