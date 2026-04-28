import pygame
import sys
import yaml

from helpers import DrawingHelper

class ZipEnvironment:
    def __init__(self):
        # Read params from YAML file
        with open("src/params.yaml", "r") as file:
            params = yaml.safe_load(file)
        
        # Set the targets (numbers to be visited in order)
        self.targets = {int(k): tuple(v) for k, v in params["targets"].items()}
        self.start = tuple(params["start_position"])

        # Check if there are blocked edges (walls)
        if "blocked_edges" in params:
            self.blockedEdges = {tuple(map(tuple, edge)) for edge in params["blocked_edges"]}
            self.hasBlockedEdges = True
        else:
            self.blockedEdges = set()
            self.hasBlockedEdges = False

        # Set grid dimensions
        self.ROWS = params["grid"]["rows"]
        self.COLS = params["grid"]["cols"]
        self.CELL = params["grid"]["cell_size"]
        self.WIDTH = self.COLS * self.CELL
        self.HEIGHT = self.ROWS * self.CELL
        
        # Initialize drawing helper
        self.helper = DrawingHelper()
        
        # Define actions (up, down, left, right)
        self.ACTIONS = [
            (-1, 0), # Up
            (1, 0), # Down
            (0, -1), # Left
            (0, 1) # Right
        ]
        
        self.isPuzzleComplete = False
        
        self.reset()

    # Get valid actions from a given position (considering grid boundaries and walls)
    def validActions(self, pos):
        valid = []

        for i, action in enumerate(self.ACTIONS):
            dr, dc = action
            newPos = (pos[0] + dr, pos[1] + dc)

            # Sai do grid?
            if not self.inside(newPos):
                continue

            # Bate em barreira?
            if self.hasBlockedEdges and (
                self.hasWall(pos, newPos) or
                self.hasWall(newPos, pos)
            ):
                continue
            
            valid.append(i)
        return valid

    # Reset the environment
    def reset(self):
        self.agentPos = self.start
        self.visited = {self.start: None}
        self.currentTarget = 1
        self.message = ""
        
        # Return the initial state (agent's starting position)
        return self.agentPos

    # Check if there is a wall between two positions
    def hasWall(self, a, b):
        return ((a, b) in self.blockedEdges or (b, a) in self.blockedEdges)

    # Check if a position is inside the grid
    def inside(self, pos):
        r, c = pos
        return 0 <= r < self.ROWS and 0 <= c < self.COLS

    # Take a step in the environment based on the action taken by the agent
    def step(self, action):
        dr, dc = action
        Sl = self.agentPos
        R = -1
        done = False

        newPos = (self.agentPos[0] + dr, self.agentPos[1] + dc)

        if self.hasReachedDeadEnd():
            self.message = "Dead end reached"
            R = -100
            done = True
            
            return [Sl, R, done]

        if newPos in self.visited:
            self.message = "Already visited"
            R = -15
            return [Sl, R, done]

        # Valid move: register move and the previous direction
        self.visited[self.agentPos] = action
        self.agentPos = newPos
        self.visited[newPos] = None
        Sl = newPos

        # The agent made a valid move and got the right target
        if (self.currentTarget in self.targets and
            newPos == self.targets[self.currentTarget]):

            self.message = f"Found {self.currentTarget}"
            self.currentTarget += 1

            # Check if the puzzle is complete (all targets found and all cells visited)
            if self.currentTarget > len(self.targets) and len(self.visited) == self.ROWS * self.COLS:
                self.message = "Puzzle complete!"
                
                # Puzzle complete
                R = 300
                done = True
                
                print("Puzzle complete!")
                self.isPuzzleComplete = True
                
                return [Sl, R, done]

            # Found the correct target but the puzzle is not complete yet
            R = 30
            return [Sl, R, done]

        # The agent made a valid move, but got the wrong target
        if newPos in self.targets.values():
            self.message = "Wrong number"
            R = -60
            return [Sl, R, done]
            
        # The agent made a valid move, but got to a dead end
        if not self.hasReachedEnd() and self.hasReachedDeadEnd():
            self.message = "Dead end reached"
            R = -100
            done = True
            
            return [Sl, R, done]
            
        R = 2 # Valid move, but no target found yet
        return [Sl, R, done]
            
    def initializeGrid(self):
        pygame.init()

        # Set up the display
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Zip Environment")
        self.font = pygame.font.SysFont(None, 30)
        
        self.clock = pygame.time.Clock()
        
    # Render the game state on the screen
    def renderGame(self, episode):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        self.screen.fill(self.helper.WHITE)

        # Draw the game elements
        self.helper.drawGrid(self.screen, self.ROWS, self.COLS, self.CELL)
        self.helper.drawTargets(self.screen, self.targets, self.CELL, self.font)
        self.helper.drawVisited(self.screen, self, self.CELL)
        self.helper.drawMsg(self.screen, self.message, self.font, episode)
        
        if self.hasBlockedEdges:
            self.helper.drawWalls(self.screen, self, self.CELL)
        
        pygame.display.flip()
        self.clock.tick(30)
        
    # Check if the agent has reached the end of the puzzle (all targets found and all cells visited)
    def hasReachedEnd(self):
        return self.currentTarget > len(self.targets) and len(self.visited) == self.ROWS * self.COLS
    
    # Check if the agent has reached a dead end (no valid moves left)
    def hasReachedDeadEnd(self):
        # Check if there are any valid moves left
        for action in self.ACTIONS:
            dr, dc = action
            newPos = (self.agentPos[0] + dr, self.agentPos[1] + dc)
            
            if (self.inside(newPos) and
                (not self.hasBlockedEdges or not self.hasWall(self.agentPos, newPos)) and
                newPos not in self.visited):
                return False  # There is at least one valid move left

        return True # No valid moves left