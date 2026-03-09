import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PaddleEnv(gym.Env):
    """
    A minimal 2D Paddle game environment for fast CPU/MPS training.
    
    State (5 dims): [ball_x, ball_y, ball_vx, ball_vy, paddle_y]
    All normalized to roughly [-1, 1].
    
    Action: Discrete 0 (up), 1 (stay), 2 (down)
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # Hyperparameters
        self.max_steps = 1000
        self.dt = 0.05
        
        self.paddle_height = 0.2
        self.paddle_speed = 0.8
        self.paddle_x = 0.9 # Paddle is on the right side
        
        # Action space: 0: up, 1: stay, 2: down
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 5 numbers
        low = np.array([-1.5, -1.5, -2.0, -2.0, -1.5], dtype=np.float32)
        high = np.array([1.5, 1.5, 2.0, 2.0, 1.5], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Random initial ball position on the left half
        self.ball_x = self.np_random.uniform(-0.8, 0.0)
        self.ball_y = self.np_random.uniform(-0.8, 0.8)
        
        # Random initial velocity towards the right
        self.ball_vx = self.np_random.uniform(0.5, 1.0)
        self.ball_vy = self.np_random.uniform(-0.8, 0.8)
        
        self.paddle_y = 0.0
        
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.ball_x,
            self.ball_y,
            self.ball_vx,
            self.ball_vy,
            self.paddle_y
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        
        # 1. Update paddle
        if action == 0:
            self.paddle_y += self.paddle_speed * self.dt
        elif action == 2:
            self.paddle_y -= self.paddle_speed * self.dt
            
        # Clamp paddle
        self.paddle_y = np.clip(self.paddle_y, -1.0 + self.paddle_height/2, 1.0 - self.paddle_height/2)
        
        # 2. Update ball
        self.ball_x += self.ball_vx * self.dt
        self.ball_y += self.ball_vy * self.dt
        
        # Bounce off top and bottom walls
        if self.ball_y >= 1.0:
            self.ball_y = 1.0 - (self.ball_y - 1.0)
            self.ball_vy *= -1
        elif self.ball_y <= -1.0:
            self.ball_y = -1.0 + (-1.0 - self.ball_y)
            self.ball_vy *= -1
            
        # Bounce off left wall (keep game going)
        if self.ball_x <= -1.0:
            self.ball_x = -1.0 + (-1.0 - self.ball_x)
            self.ball_vx *= -1
            
        # 3. Check for paddle collision or miss
        reward = 0.0
        terminated = False
        truncated = False
        
        if self.ball_x >= self.paddle_x:
            # Ball reached paddle's x line
            if abs(self.ball_y - self.paddle_y) <= self.paddle_height / 2:
                # HIT!
                reward = 1.0
                # Bounce back
                self.ball_x = self.paddle_x - (self.ball_x - self.paddle_x)
                self.ball_vx *= -1
                # Small random variation to make it interesting
                self.ball_vy += self.np_random.uniform(-0.2, 0.2)
            else:
                # MISS!
                reward = -1.0
                terminated = True
                
        # Better shaped reward for keeping paddle aligned with ball y
        # We give a positive reward when aligned, negative when far away
        y_dist = abs(self.paddle_y - self.ball_y)
        reward += 0.05 * (0.5 - y_dist)
        
        # Reward tightly tracking the ball (reduces jitter)
        if y_dist < 0.1:
            reward += 0.05
            
        if self.current_step >= self.max_steps:
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return
            
        try:
            import pygame
        except ImportError:
            print("Please install pygame to render: pip install pygame")
            return
            
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((400, 400))
            pygame.display.set_caption("Paddle Game RL")
            self.clock = pygame.time.Clock()
            
        # Draw background
        self.screen.fill((0, 0, 0))
        
        # Convert coords from [-1, 1] to [0, 400]
        def to_px(x, y):
            return int((x + 1.0) / 2.0 * 400), int((y + 1.0) / 2.0 * 400)
            
        # Draw ball
        bx, by = to_px(self.ball_x, self.ball_y)
        pygame.draw.circle(self.screen, (255, 255, 255), (bx, by), 5)
        
        # Draw paddle
        px_x, px_y_center = to_px(self.paddle_x, self.paddle_y)
        _, px_h = to_px(-1.0, -1.0 + self.paddle_height)
        
        paddle_rect = pygame.Rect(
            px_x - 5, 
            px_y_center - px_h // 2, 
            10, 
            px_h
        )
        pygame.draw.rect(self.screen, (255, 255, 255), paddle_rect)
        
        pygame.display.flip()
        self.clock.tick(60) # 60 FPS
        
        # Handle quit event to avoid unresponsiveness
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

