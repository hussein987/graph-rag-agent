from langchain_core.tools import tool
from src.utils.config import config

# Initialize Tavily API key
_tavily_api_key = config.tavily_api_key


def _get_mock_weather(location: str) -> str:
    """Return mock weather data when Tavily API key is not available."""
    return f"""Current Weather in {location}:
🌡️ Temperature: 22°C (feels like 24°C)
🌤️ Condition: Partly Cloudy
💧 Humidity: 65%
💨 Wind Speed: 3.2 m/s

⚠️ Note: This is mock data. Add TAVILY_API_KEY to get real weather information."""


@tool
def get_weather(location: str) -> str:
    """
    Get current weather for a location using Tavily search.

    Args:
        location: City name or location (e.g., "Warsaw", "London", "New York")

    Returns:
        Formatted weather information string
    """
    if not _tavily_api_key:
        return _get_mock_weather(location)

    try:
        # Import TavilySearch here to avoid import issues if not installed
        from langchain_tavily import TavilySearch

        # Initialize Tavily search tool
        search_tool = TavilySearch(
            api_key=_tavily_api_key,
            max_results=3,
            topic="general",
            search_depth="basic",
        )

        # Create weather search query
        query = f"current weather in {location} today temperature humidity wind"

        # Perform the search
        search_result = search_tool.invoke({"query": query})

        # Return the raw search result for the response formatter to handle
        return search_result

    except ImportError:
        return f"Tavily search not available. Please install langchain-tavily: pip install langchain-tavily"
    except Exception as e:
        return f"Weather service temporarily unavailable: {str(e)}"


# Export tools for easy import
__all__ = ["get_weather"]
