import pygame
import sys
import yaml

from Drawing import DrawingHelper

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
        return (a, b) in self.blockedEdges

    def inside(self, pos):
        r, c = pos
        return 0 <= r < self.ROWS and 0 <= c < self.COLS

    def step(self, action):
        dr, dc = action

        newPos = (self.agentPos[0] + dr, self.agentPos[1] + dc)

        # The agent attempted to leave the grid
        if not self.inside(newPos):
            self.message = "Invalid move"
            return

        # The agent hit a wall (from any direction)
        if self.hasBlockedEdges and (self.hasWall(self.agentPos, newPos) or self.hasWall(newPos, self.agentPos)):
            self.message = "There is a barrier"
            return

        # The agent tried to revisit a cell
        if newPos in self.visited:
            self.message = "Cell can't be revisited"
            return

        # Valid move: register move and the previous direction
        self.visited[self.agentPos] = action
        self.agentPos = newPos
        self.visited[newPos] = None

        # The agent got the right target
        if (self.currentTarget in self.targets and
            newPos == self.targets[self.currentTarget]):

            self.message = f"Found {self.currentTarget}"
            self.currentTarget += 1

            if self.currentTarget > len(self.targets) and len(self.visited) == self.ROWS * self.COLS:
                self.message = "Puzzle complete!"

        # The agent got the wrong target
        elif newPos in self.targets.values():
            self.message = "Wrong number"
            
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
                sys.exit()

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