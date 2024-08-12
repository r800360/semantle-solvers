import gym
from gym import spaces
import numpy as np
from src.data import word2vec

class SemantleEnv(gym.Env):
    def __init__(self, target_word, word_list):
        super(SemantleEnv, self).__init__()
        
        # Define the action and observation space
        self.action_space = spaces.Discrete(len(word_list))  # Assume each word is a possible action
        self.observation_space = spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32)  # Similarity score

        self.target_word = target_word
        self.word_list = word_list
        self.current_state = None  # Current similarity score

    def compute_similarity(self, word1, word2):
        x = word2vec[word1]
        y = word2vec[word2]
        return 100 * (np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y) ))

    def step(self, action):
        guessed_word = self.word_list[action]
        similarity = self.compute_similarity(guessed_word, self.target_word)
        self.current_state = similarity
        
        done = similarity == 100  # Episode is done if the similarity is perfect
        reward = similarity  # Reward is the similarity score

        return np.array([self.current_state], dtype=np.float32), reward, done, {}

    def reset(self):
        # Reset the state to an initial condition
        self.current_state = 0  # You may choose a different initial state
        return np.array([self.current_state], dtype=np.float32)

    def render(self, mode='human'):
        print(f"Current similarity: {self.current_state}")

    def close(self):
        pass
