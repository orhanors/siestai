"""
Rate limiter utility for controlling API request rates.
"""

import asyncio
import time
from typing import Dict, Optional
from dataclasses import dataclass
from app.utils.logger import get_logger

logger = get_logger("siestai.rate_limiter")

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_second: float = 1.0
    burst_size: int = 5
    
class RateLimiter:
    """Token bucket rate limiter for API requests."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = float(config.burst_size)
        self.last_update = time.time()
        self._lock = asyncio.Lock()
        
    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.time()
            # Add tokens based on time elapsed
            time_passed = now - self.last_update
            self.tokens = min(
                self.config.burst_size,
                self.tokens + time_passed * self.config.requests_per_second
            )
            self.last_update = now
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            
            # Need to wait for more tokens
            wait_time = (1.0 - self.tokens) / self.config.requests_per_second
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for next request")
            await asyncio.sleep(wait_time)
            self.tokens = 0.0

class GlobalRateLimiter:
    """Global rate limiter manager for different sources."""
    
    def __init__(self):
        self._limiters: Dict[str, RateLimiter] = {}
        
    def get_limiter(self, source: str, config: Optional[RateLimitConfig] = None) -> RateLimiter:
        """Get or create a rate limiter for a source."""
        if source not in self._limiters:
            if config is None:
                # Default configuration - conservative rate limiting
                config = RateLimitConfig(requests_per_second=0.5, burst_size=3)
            self._limiters[source] = RateLimiter(config)
            logger.info(f"Created rate limiter for {source}: {config.requests_per_second} req/s, burst: {config.burst_size}")
        return self._limiters[source]

# Global instance
global_rate_limiter = GlobalRateLimiter()