import os
from typing import Any, Dict

import httpx

OPENWEATHER_BASE_URL="https://api.openweathermap.org/data/2.5/weather"

async def fetch_weather(city: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return {"error": "Open Weather environment variable not found."}
    
    params: Dict[str, str] = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    try:
        async with httpx.AsyncClient(timeout = 10.0) as client:
            response = await client.get(OPENWEATHER_BASE_URL, params = params)

            if response.status_code == 401:
                return {
                    "error": "OpenWeather API key is unauthorized."
                }
            
            if response.status_code == 404:
                return {
                    "error": f"City '{city}' was not found."
                }
            
            response.raise_for_status()
            data = response.json()

    except httpx.TimeoutException:
        return {
            "error": f"OpenWeather request timed out for city: '{city}'"
        }
    except httpx.HTTPStatusError as ex:
        return {
            "error": f"HTTP {exec.response.status_code} error while fetching weather for city: '{city}'"
        }
    except Exception as exc:
        return {
            "error": f"Unexpected error fetching weather for city: '{city}'"
        }
    
    return {
        "city": data["name"],
        "country": data["sys"]["country"],
        "temperature_celcius": data["main"]["temp"],
        "feels_like_celcius": data["main"]["feels_like"],
        "humidity_percent": data["main"]["humidity"],
        "description": data["weather"][0]["description"],
        "wind_speed_mps": data["wind"]["speed"]
    }