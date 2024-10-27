from typing import Optional
import pygame
import gym
from gym import spaces
import numpy as np
from gym.core import ActType, ObsType, RenderFrame


class PingPongEnv(gym.Env):
    def __init__(self):
        super(PingPongEnv, self).__init__()
        # Init Attributes
        self.paddle_y = 0
        self.ball_x = 0
        self.ball_y = 0
        self.ball_velocity = np.array([0, 0])
        self.done = False
        # Game settings
        self.width = 640
        self.height = 480
        self.paddle_width = 10
        self.paddle_height = 60
        self.ball_size = 10
        self.paddle_speed = 5
        self.ball_speed = 5
        self.max_steps = 500
        self.current_step = 0

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Ping Pong RL")

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Stay, 1: Up, 2: Down
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

        # Initialize game state
        self.seed_value = None
        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[ObsType, dict]:
        if seed is not None:
            self.seed_value = seed
            np.random.seed(self.seed_value)

        self.paddle_y = self.height // 2 - self.paddle_height // 2
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_velocity = np.array([self.ball_speed, np.random.choice([-1, 1]) * self.ball_speed])

        self.done = False
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, agent_action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        if agent_action == 1:
            self.paddle_y = max(0, self.paddle_y - self.paddle_speed)
        elif agent_action == 2:
            self.paddle_y = min(self.height - self.paddle_height, self.paddle_y + self.paddle_speed)

        self.ball_x += self.ball_velocity[0]
        self.ball_y += self.ball_velocity[1]

        # Check for collisions with the top and bottom walls
        if self.ball_y <= 0 or self.ball_y >= self.height - self.ball_size:
            self.ball_velocity[1] = -self.ball_velocity[1]  # Invert y velocity

        agent_reward = 0.0

        # Check for collisions with the paddle
        if (self.ball_x <= self.paddle_width and
            self.paddle_y <= self.ball_y <= self.paddle_y + self.paddle_height):
            self.ball_velocity[0] = abs(self.ball_velocity[0])  # Ensure it moves right
            agent_reward = 1.0
        elif self.ball_x <= 0:
            self.done = True
            agent_reward = -1.0  # Penalty for missing the ball
        elif self.ball_x >= self.width - self.ball_size:
            self.ball_velocity[0] = -abs(self.ball_velocity[0])  # Ensure it moves left

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        return self._get_obs(), agent_reward, self.done, False, {}

    def _get_obs(self) -> ObsType:
        return np.array([
            self.paddle_y / self.height,
            self.ball_x / self.width,
            self.ball_y / self.height,
            self.ball_velocity[0] / self.ball_speed,
            self.ball_velocity[1] / self.ball_speed
        ], dtype=np.float32)

    def render(self, mode='human') -> Optional[RenderFrame]:
        for game_event in pygame.event.get():
            if game_event.type == pygame.QUIT:
                self.close()

        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), (0, self.paddle_y, self.paddle_width, self.paddle_height))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.ball_x, self.ball_y, self.ball_size, self.ball_size))

        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.current_step}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.width - 150, 10))

        pygame.display.flip()
        return None

    def close(self) -> None:
        pygame.quit()
        exit()


if __name__ == "__main__":
    env = PingPongEnv()
    obs, _ = env.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        env.render()
        pygame.time.delay(30)

    env.close()
