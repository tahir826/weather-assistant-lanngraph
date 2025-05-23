from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import chainlit as cl
import os
import requests
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# API Keys validation
w_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
if not w_api_key:
    raise ValueError("OPENWEATHERMAP_API_KEY is not set. Please ensure it is defined in your .env file.")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

weatherapi_key = os.getenv("WEATHERAPI_KEY")
if not weatherapi_key:
    raise ValueError("WEATHERAPI_KEY is not set. Please ensure it is defined in your .env file.")

tomorrow_key = os.getenv("TOMORROW_KEY")
if not tomorrow_key:
    raise ValueError("TOMORROW_KEY is not set. Please ensure it is defined in your .env file.")

visualcrossing_key = os.getenv("VISUALCROSSING_KEY")
if not visualcrossing_key:
    raise ValueError("VISUALCROSSING_KEY is not set. Please ensure it is defined in your .env file.")

# Weather Tools
def get_current_weather(city: str) -> dict:
    """
    Fetch current weather information for a specific city using OpenWeatherMap.
    Use this tool for current weather queries.
    
    Args:
        city (str): Location city or place name.
    Returns:
        dict: Current weather data with temperature, description, humidity, and wind speed.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={w_api_key}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        weather = {
            "source": "OpenWeatherMap",
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "pressure": data["main"]["pressure"]
        }
        return weather
    else:
        return {"error": f"Failed to get current weather data: {response.status_code}"}

def get_daily_forecast(city: str, days: int = 3) -> dict:
    """
    Fetch multi-day weather forecast using WeatherAPI.
    Use this tool for daily forecast queries (3-10 days).
    
    Args:
        city (str): Location city or place name.
        days (int): Number of days for the forecast (1-10, default is 3).
    Returns:
        dict: Daily weather forecast data.
    """
    # Limit days to maximum 10
    days = min(max(days, 1), 10)
    
    url = f"http://api.weatherapi.com/v1/forecast.json?key={weatherapi_key}&q={city}&days={days}&aqi=no&alerts=no"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        forecast = [
            {
                "date": day["date"],
                "condition": day["day"]["condition"]["text"],
                "max_temp": day["day"]["maxtemp_c"],
                "min_temp": day["day"]["mintemp_c"],
                "avg_temp": day["day"]["avgtemp_c"],
                "humidity": day["day"]["avghumidity"],
                "wind_kph": day["day"]["maxwind_kph"],
                "rain_chance": day["day"]["daily_chance_of_rain"]
            }
            for day in data.get("forecast", {}).get("forecastday", [])
        ]
        return {
            "source": "WeatherAPI",
            "location": data.get("location", {}).get("name", city),
            "country": data.get("location", {}).get("country", ""),
            "forecast_type": "daily",
            "forecast": forecast
        }
    else:
        return {"error": f"WeatherAPI daily forecast error: {response.status_code}"}

def get_hourly_forecast(city: str, hours: int = 24) -> dict:
    """
    Fetch hourly weather forecast using Tomorrow.io API.
    Use this tool for hourly weather forecast queries (up to 120 hours).
    
    Args:
        city (str): Location city or place name.
        hours (int): Number of hours for the forecast (1-120, default is 24).
    Returns:
        dict: Hourly weather forecast data.
    """
    # Limit hours to maximum 120 (5 days)
    hours = min(max(hours, 1), 120)
    
    url = (
        f"https://api.tomorrow.io/v4/weather/forecast"
        f"?location={city}"
        f"&apikey={tomorrow_key}"
        f"&timesteps=1h"
        f"&units=metric"
        f"&fields=temperature,windSpeed,humidity,precipitationProbability,weatherCode"
        f"&startTime=now"
        f"&endTime=nowPlus{hours}h"
    )
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        timelines = data.get("timelines", {}).get("hourly", [])
        
        forecast = [
            {
                "time": entry["time"],
                "temperature": entry["values"].get("temperature"),
                "wind_speed": entry["values"].get("windSpeed"),
                "humidity": entry["values"].get("humidity"),
                "precipitation_chance": entry["values"].get("precipitationProbability"),
                "weather_code": entry["values"].get("weatherCode")
            }
            for entry in timelines
        ]
        
        return {
            "source": "Tomorrow.io",
            "location": city,
            "forecast_type": "hourly",
            "forecast": forecast
        }
    else:
        return {"error": f"Tomorrow.io hourly forecast error: {response.status_code}"}

def get_historical_weather(city: str, start_date: str, end_date: str = None) -> dict:
    """
    Fetch historical weather data using Visual Crossing API.
    Use this tool for historical weather queries and past weather information.
    
    Args:
        city (str): Location city or place name.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format (optional, defaults to start_date).
    Returns:
        dict: Historical weather data.
    """
    if not end_date:
        end_date = start_date
    
    base_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/{start_date}/{end_date}"
    params = {
        "key": visualcrossing_key,
        "unitGroup": "metric",
        "include": "days"
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        location = data.get("resolvedAddress", city)
        
        forecast = [
            {
                "date": day.get("datetime"),
                "max_temp": day.get("tempmax"),
                "min_temp": day.get("tempmin"),
                "avg_temp": day.get("temp"),
                "description": day.get("conditions"),
                "humidity": day.get("humidity"),
                "wind_kph": day.get("windspeed"),
                "precipitation": day.get("precip")
            }
            for day in data.get("days", [])
        ]
        
        return {
            "source": "Visual Crossing",
            "location": location,
            "forecast_type": "historical",
            "forecast": forecast
        }
    else:
        return {"error": f"Visual Crossing historical weather error: {response.status_code}"}

# Initialize LLM and Graph
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)
tools = [get_current_weather, get_daily_forecast, get_hourly_forecast, get_historical_weather]
llm_with_tools = llm.bind_tools(tools)

def tool_calling_llm(state: MessagesState) -> MessagesState:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_node("final_response", tool_calling_llm)  # Add final response node

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm", 
    tools_condition,
    {
        "tools": "tools",
        END: END
    }
)
builder.add_edge("tools", "final_response")  # After tools, generate final response
builder.add_edge("final_response", END)

graph = builder.compile()

# System message for the assistant
SYSTEM_MESSAGE = """You are a helpful weather assistant that provides accurate weather information. 

Guidelines:
1. Use the appropriate tool based on the user's query:
   - get_current_weather: For current weather conditions
   - get_daily_forecast: For daily forecasts (3-10 days)
   - get_hourly_forecast: For hourly forecasts (up to 120 hours)
   - get_historical_weather: For past weather data

2. Always provide weather information in a clear, conversational format
3. Include relevant details like temperature, conditions, humidity, and wind
4. If there's an error, explain it clearly and suggest alternatives
5. Be concise but informative in your responses
"""

# Chainlit Integration
@cl.on_chat_start
async def start():
    await cl.Message(
        content="🌤️ Welcome to Weather Assistant! Ask me about current weather, forecasts, or historical weather data for any city.",
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # Show typing indicator
    async with cl.Step(name="AI Response", type="tool") as step:
        step.output = "Generating Output..."
        
        # Prepare messages for the graph
        messages = [
            SystemMessage(content=SYSTEM_MESSAGE),
            HumanMessage(content=message.content)
        ]
        
        try:
            # Invoke the graph
            result = graph.invoke({"messages": messages})
            
            # Extract only the final AI response
            response_content = extract_final_ai_response(result)
            
            if response_content:
                step.output = "Response received."
            else:
                response_content = "I apologize, but I couldn't retrieve the weather information. Please try again with a different query."
                step.output = "Failed to retrieve weather data."
                
        except Exception as e:
            response_content = f"I encountered an error while fetching weather data: {str(e)}"
            step.output = f"Error: {str(e)}"
    
    # Send only the AI response
    await cl.Message(content=response_content).send()

def extract_final_ai_response(result):
    """
    Extract the final AI response from the graph result.
    Handles cases where AI response comes after tool calls.
    """
    messages = result.get("messages", [])
    
    # Find all AI messages
    ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
    
    if not ai_messages:
        return None
    
    # Get the last AI message
    last_ai_message = ai_messages[-1]
    
    # If the last AI message has content, return it
    if last_ai_message.content and last_ai_message.content.strip():
        return last_ai_message.content
    
    # If the last AI message is a tool call, we need the AI response after tool execution
    # Look for the final AI message that comes after all tool messages
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
            return msg.content
    
    return None

if __name__ == "__main__":
    # For testing without Chainlit
    test_messages = [
        SystemMessage(content=SYSTEM_MESSAGE),
        HumanMessage(content="What's the current weather in London?")
    ]
    
    result = graph.invoke({"messages": test_messages})
    ai_response = extract_final_ai_response(result)
    print("AI Response:", ai_response)
    
    # Debug: Print the structure to understand the message flow
    print("\nMessage structure:")
    for i, msg in enumerate(result["messages"]):
        print(f"{i}: {type(msg).__name__} - Content: {msg.content[:100] if msg.content else 'No content'}")
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"   Tool calls: {[tc['name'] for tc in msg.tool_calls]}")