from typing import Dict, List, Tuple
from collections import defaultdict

class EloRatingSystem:
    """
    Implements a standard pairwise Elo rating system to rank models based on Head-to-Head performance.
    """
    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1200.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        # defaultdict automatically initializes new models with the starting rating
        self.ratings: Dict[str, float] = defaultdict(lambda: initial_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate the expected probability of Model A winning against Model B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(self, player_a: str, player_b: str, actual_score_a: float):
        """
        Update ratings based on match outcome.
        actual_score_a: 1.0 if A wins, 0.5 for draw, 0.0 if B wins
        """
        ra = self.ratings[player_a]
        rb = self.ratings[player_b]
        
        expected_a = self.expected_score(ra, rb)
        expected_b = self.expected_score(rb, ra)
        
        actual_score_b = 1.0 - actual_score_a
        
        self.ratings[player_a] = ra + self.k_factor * (actual_score_a - expected_a)
        self.ratings[player_b] = rb + self.k_factor * (actual_score_b - expected_b)

    def process_matchups(self, matchups: List[Tuple[str, str, float]]):
        """
        Process a batch of matchups. 
        Format: [(model_a, model_b, score_a_vs_b)]
        """
        for player_a, player_b, score_a in matchups:
            self.update_ratings(player_a, player_b, score_a)
            
    def get_leaderboard(self) -> List[Tuple[str, float]]:
        """Return the sorted leaderboard."""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
