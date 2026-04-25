import yaml
from environment import ZipEnvironment

# Read params from YAML file
with open("src/params.yaml", "r") as file:
    params = yaml.safe_load(file)
    
if __name__ == "__main__":
    env = ZipEnvironment()
    env.initializeGrid()

    while True:
        env.renderGame()