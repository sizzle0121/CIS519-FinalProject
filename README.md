# Deep Flappy Bird
A Crazy Flappy Bird agent trained with Deep Reinforcement Learning 
> Support training on multiple platform     
> e.g., Amazon EC2, Amazon SageMaker Notebook, Google Cloud Compute Engine, Google Colab

### Contributors
> Name: Fu-Lin Hsu     
> Name: James Hu    
> Name: Jonathan Choi

### Requirement
- Python 3.x
- Pygame
- Pytorch
- Numpy
- Opencv 3.x or later

### Before you start
1. Be sure you upload game assets in the directory order './assets/sprites/*.png'   
    + you can set the BASE_PATH as the prefix of assets/sprites/*.png, such as a mounted drive in Colab
    + Remember to end it with a slash '/'
2. Set up the directory you want to save checkpoints

### Environment

#### env.py
The Flappy Bird Game Environment    
To load resources (images) for the game, specify the __BASE_PATH__ parameter of __load()__  
Use __frame_step(input_actions)__ to interact with the game and get the next observation    
The game ends when the return value, __terminal__, of frame_step is True

#### env_utils.py
The utilities for establishing the environment

#### flappy_bird.py
The all-in-one training instance including game environment

