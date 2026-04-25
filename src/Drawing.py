import pygame
import yaml

class DrawingHelper:
    
    def __init__(self):
        # Read params from YAML file
        with open("src/params.yaml", "r") as file:
            params = yaml.safe_load(file)
    
        # Set colors
        self.WHITE = params["grid"]["white"]
        self.BLACK = params["grid"]["black"]
        self.GRAY = params["grid"]["gray"]
        self.GREEN = params["grid"]["green"]
        self.RED = params["grid"]["red"]
    
    def drawGrid(self, screen, rows, cols, cellSize):
        for r in range(rows):
            for c in range(cols):
                rect = pygame.Rect(c * cellSize, r * cellSize, cellSize, cellSize)
                pygame.draw.rect(screen, self.GRAY, rect, 1)
    
    def drawWalls(self, screen, env, cellSize):
        for edge in env.blockedEdges:
            a, b = edge

            r1, c1 = a
            r2, c2 = b

            # Only draw once
            if (r1, c1) > (r2, c2):
                continue

            # Vertical wall
            if r1 == r2:
                x = max(c1, c2) * cellSize
                y1 = r1 * cellSize
                y2 = y1 + cellSize

                pygame.draw.line(screen, self.RED, (x, y1), (x, y2), 5)

            # Horizontal wall
            elif c1 == c2:
                y = max(r1, r2) * cellSize
                x1 = c1 * cellSize
                x2 = x1 + cellSize

                pygame.draw.line(screen, self.RED, (x1, y), (x2, y), 5)
    
    def drawVisited(self, screen, env, cellSize):
        for cell, direction in env.visited.items():
            r, c = cell

            cx = c * cellSize + cellSize // 2
            cy = r * cellSize + cellSize // 2

            # Background rectangle
            rect = pygame.Rect(
                c * cellSize + 15, r * cellSize + 15, cellSize - 30, cellSize - 30
            )
            
            pygame.draw.rect(screen, self.GREEN, rect)

            if direction is None:
                continue

            length = 12
            halfBase = 8

            # Draw an arrow pointing in the direction of the previous move
            if direction == (0,1): # →
                pygame.draw.polygon(
                    screen,
                    self.BLACK,
                    [
                        (cx + length, cy),
                        (cx - halfBase, cy - halfBase),
                        (cx - halfBase, cy + halfBase)
                    ]
                )

            elif direction == (0,-1): # ←
                pygame.draw.polygon(
                    screen,
                    self.BLACK,
                    [
                        (cx - length, cy),
                        (cx + halfBase, cy - halfBase),
                        (cx + halfBase, cy + halfBase)
                    ]
                )

            elif direction == (-1,0): # ↑
                pygame.draw.polygon(
                    screen,
                    self.BLACK,
                    [
                        (cx, cy - length),
                        (cx - halfBase, cy + halfBase),
                        (cx + halfBase, cy + halfBase)
                    ]
                )

            elif direction == (1,0): # ↓
                pygame.draw.polygon(
                    screen,
                    self.BLACK,
                    [
                        (cx, cy + length),
                        (cx - halfBase, cy - halfBase),
                        (cx + halfBase, cy - halfBase)
                    ]
                )
    
    def drawTargets(self, screen, targets, cellSize, font):
        for n, pos in targets.items():
            r, c = pos

            center = (
                c * cellSize + cellSize // 2,
                r * cellSize + cellSize // 2
            )

            # Draw a circle with the target number inside
            pygame.draw.circle(screen, self.BLACK, center, 22)

            txt = font.render(str(n), True, self.WHITE)
            txtRect = txt.get_rect(center = center)
            
            screen.blit(txt, txtRect)
            
    def drawMsg(self, screen, msg, font):
        msgRender = font.render(msg, True, self.BLACK)
        screen.blit(msgRender, (10, 10))