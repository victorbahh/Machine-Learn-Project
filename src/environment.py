import pygame
import sys
import yaml

from .drawing import DrawingHelper


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
        self.ACTIONS = {
            pygame.K_UP: (-1, 0),
            pygame.K_DOWN: (1, 0),
            pygame.K_LEFT: (0, -1),
            pygame.K_RIGHT: (0, 1)
        }
        
        self.reset()

    def reset(self):
        self.agentPos = self.start
        self.visited = {self.start: None}
        self.currentTarget = 1
        self.message = ""

    def hasWall(self, a, b):
        return ((a, b) in self.blockedEdges or (b, a) in self.blockedEdges)

    def inside(self, pos):
        r, c = pos
        return 0 <= r < self.ROWS and 0 <= c < self.COLS

    def step(self, action):
        dr, dc = action
        # Reward for valid move
        reward = 0.1  
        done = False

        newPos = (self.agentPos[0] + dr, self.agentPos[1] + dc)

        # Agent attempted to move outside the grid
        if not self.inside(newPos):
            self.message = "Invalid move"
            return self.agentPos, -1, False, {}

        # Agent hit a wall (blocked edge)
        if self.hasBlockedEdges and self.hasWall(self.agentPos, newPos):
            self.message = "There is a barrier"
            return self.agentPos, -1, False, {}

        # Agent attempted to revisit a cell
        if newPos in self.visited:
            self.message = "Cell can't be revisited"
            return self.agentPos, -1, False, {}

        # Valid move: update position and visited path
        self.visited[self.agentPos] = action
        self.agentPos = newPos
        self.visited[newPos] = None

        # Agent reached the correct target in sequence
        if (self.currentTarget in self.targets and
            newPos == self.targets[self.currentTarget]):

            reward = 10
            self.message = f"Found {self.currentTarget}"
            self.currentTarget += 1

            # Check if puzzle is fully completed
            if self.hasReachedEnd():
                reward = 50
                done = True
                self.message = "Puzzle complete!"

        # Agent reached a target but in the wrong order
        elif newPos in self.targets.values():
            reward = -5
            self.message = "Wrong number"

        # Check if no valid moves are left (dead end)
        if self.hasReachedDeadEnd():
            done = True
            reward = -10
            self.message = "Dead end"

        self.message = "Reward: " + str(reward)
        return self.agentPos, reward, done, {}   
    
    def initializeGrid(self):
        pygame.init()

        # Set up the display
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Zip Environment")
        self.font = pygame.font.SysFont(None, 30)
        
        self.clock = pygame.time.Clock()
        
    def renderGame(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.KEYDOWN:
                if event.key in self.ACTIONS:
                    self.step(
                        self.ACTIONS[event.key]
                    )

                if event.key == pygame.K_r:
                    self.reset()
        
        self.screen.fill(self.helper.WHITE)

        # Draw the game elements
        self.helper.drawGrid(self.screen, self.ROWS, self.COLS, self.CELL)
        self.helper.drawTargets(self.screen, self.targets, self.CELL, self.font)
        self.helper.drawVisited(self.screen, self, self.CELL)
        self.helper.drawMsg(self.screen, self.message, self.font)
        
        if self.hasBlockedEdges:
            self.helper.drawWalls(self.screen, self, self.CELL)
        
        pygame.display.flip()
        self.clock.tick(30)

    def closeRender(self):
        pygame.display.quit()
        pygame.quit()
        
    def hasReachedEnd(self):
        return self.currentTarget > len(self.targets) and len(self.visited) == self.ROWS * self.COLS
    
    def hasReachedDeadEnd(self):
        # Check if there are any valid moves left
        for action in self.ACTIONS.values():
            dr, dc = action
            newPos = (self.agentPos[0] + dr, self.agentPos[1] + dc)

            if (self.inside(newPos) and
                (not self.hasBlockedEdges or not self.hasWall(self.agentPos, newPos)) and
                newPos not in self.visited):
                return False  # There is at least one valid move left

        return True  # No valid moves left