"""
Rate limiting module.
Implements in-memory rate limiting per user/API key.
"""
import time
from collections import defaultdict, deque
from typing import Dict, Deque
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class RateLimiter:
    """
    In-memory rate limiter using sliding window.
    Tracks requests per user within a time window.
    """
    
    def __init__(self, max_requests: int = None, window_seconds: int = 60):
        """
        Args:
            max_requests: Maximum requests allowed in window (from config if None)
            window_seconds: Time window in seconds (default 60 for RPM)
        """
        self.max_requests = max_requests or config.RATE_LIMIT_RPM
        self.window_seconds = window_seconds
        
        # user_id -> deque of timestamps
        self.request_log: Dict[int, Deque[float]] = defaultdict(deque)
    
    def is_allowed(self, user_id: int) -> bool:
        """
        Check if a request from user_id is allowed.
        Also cleans up old requests from the window.
        
        Returns:
            True if request is allowed, False if rate limited
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # Get user's request log
        user_log = self.request_log[user_id]
        
        # Remove requests outside the current window
        while user_log and user_log[0] < window_start:
            user_log.popleft()
        
        # Check if under limit
        if len(user_log) < self.max_requests:
            # Allow request and log it
            user_log.append(now)
            return True
        else:
            # Rate limited
            return False
    
    def get_remaining(self, user_id: int) -> int:
        """Get remaining requests for a user in current window."""
        now = time.time()
        window_start = now - self.window_seconds
        
        user_log = self.request_log[user_id]
        
        # Count requests in current window
        valid_requests = sum(1 for ts in user_log if ts >= window_start)
        
        return max(0, self.max_requests - valid_requests)
    
    def reset_user(self, user_id: int):
        """Reset rate limit for a specific user."""
        if user_id in self.request_log:
            del self.request_log[user_id]


# Global rate limiter instance
rate_limiter = RateLimiter()
