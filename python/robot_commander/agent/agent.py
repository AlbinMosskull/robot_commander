
class Agent:
    def __init__(self, x, y, v):
        self.x = x
        self.y = y
        self.v = v

    def move(self, goal_x, goal_y):
        error_x = goal_x - self.x
        error_y = goal_y - self.y
        distance = (error_x**2 + error_y**2)**0.5

        if distance < 1e-5:
            return

        v = min(self.v, distance)
        self.x += v * (error_x / distance)
        self.y += v * (error_y / distance)
