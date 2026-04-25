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
        if env.hasReachedEnd():
            time.sleep(0.5) # Pause for a moment before quitting
            break
        
        if env.hasReachedDeadEnd():
            time.sleep(0.5) # Pause for a moment before resetting
            env.reset()
        
        env.renderGame()