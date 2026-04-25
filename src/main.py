import time
import yaml
from environment import ZipEnvironment

# Read params from YAML file
with open("src/params.yaml", "r") as file:
    params = yaml.safe_load(file)
    
if __name__ == "__main__":
    env = ZipEnvironment()
    env.initializeGrid()

    while True:
        # Agent has solved the puzzle
        if env.hasReachedEnd():
            time.sleep(0.5)
            break
        
        # Agent has reached a terminal state (end of episode)
        if env.hasReachedDeadEnd():
            time.sleep(0.5)
            env.reset()
        
        env.renderGame()