import os
import requests
from datetime import datetime
from typing import Dict, Any, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from dotenv import load_dotenv
import chainlit as cl
from typing import Annotated, TypedDict

# Load environment variables
load_dotenv()

# Validate API keys
w_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
if not w_api_key:
    raise ValueError("OPENWEATHERMAP_API_KEY is not set.")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set.")
weatherapi_key = os.getenv("WEATHERAPI_KEY")
if not weatherapi_key:
    raise ValueError("WEATHERAPI_KEY is not set.")
tomorrow_key = os.getenv("TOMORROW_KEY")
if not tomorrow_key:
    raise ValueError("TOMORROW_KEY is not set.")
visualcrossing_key = os.getenv("VISUALCROSSING_KEY")
if not visualcrossing_key:
    raise ValueError("VISUALCROSSING_KEY is not set.")

# Initialize the model (using LangChain's ChatOpenAI with custom endpoint for Gemini)
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=api_key
)

# Define tools
@tool
def get_weatherapi_forecast(city: str, days: int = 3) -> dict:
    """Fetch weather forecast from WeatherAPI for a specific city."""
    url = f"http://api.weatherapi.com/v1/forecast.json?key={weatherapi_key}&q={city}&days={days}&aqi=no&alerts=no"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        forecast = [
            {
                "date": day["date"],
                "condition": day["day"]["condition"]["text"],
                "avg_temp": day["day"]["avgtemp_c"],
                "humidity": day["day"]["avghumidity"],
                "wind_kph": day["day"]["maxwind_kph"]
            }
            for day in data.get("forecast", {}).get("forecastday", [])
        ]
        return {
            "source": "WeatherAPI",
            "location": data.get("location", {}).get("name", city),
            "forecast": forecast
        }
    else:
        return {"error": f"WeatherAPI error: {response.status_code}"}

@tool
def get_tomorrow_forecast(city: str, hours: int = 24) -> dict:
    """Fetch weather forecast from Tomorrow.io for a specific city."""
    url = (
        f"https://api.tomorrow.io/v4/weather/forecast"
        f"?location={city}"
        f"&apikey={tomorrow_key}"
        f"&timesteps=1h"
        f"&units=metric"
        f"&limit={hours}"
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
                "precipitation": entry["values"].get("precipitationProbability"),
            }
            for entry in timelines
        ]
        return {"source": "Tomorrow.io", "location": city, "forecast": forecast}
    else:
        return {"error": f"Tomorrow.io error: {response.status_code}"}

@tool
def get_visualcrossing_weather(
    city: str,
    start_date: str = None,
    end_date: str = None,
    include_hours: bool = False,
    days_limit: int = 3
) -> dict:
    """Fetch weather data from Visual Crossing for a specific city."""
    base_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}"
    params = {
        "key": visualcrossing_key,
        "unitGroup": "metric",
        "include": "hours" if include_hours else "days"
    }
    
    if start_date:
        base_url += f"/{start_date}"
    if end_date:
        base_url += f"/{end_date}"
    elif start_date:
        base_url += f"/{start_date}"
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        location = data.get("resolvedAddress", city)
        
        if include_hours:
            forecast = []
            for day in data.get("days", []):
                for hour in day.get("hours", []):
                    forecast.append({
                        "datetime": hour.get("datetime"),
                        "temp": hour.get("temp"),
                        "description": hour.get("conditions"),
                        "humidity": hour.get("humidity"),
                        "wind_kph": hour.get("windspeed"),
                    })
        else:
            forecast = [
                {
                    "date": day.get("datetime"),
                    "temp": day.get("temp"),
                    "description": day.get("conditions"),
                    "humidity": day.get("humidity"),
                    "wind_kph": day.get("windspeed"),
                }
                for day in data.get("days", [])[:days_limit]
            ]
        
        return {"source": "Visual Crossing", "location": location, "forecast": forecast}
    else:
        return {"error": f"Visual Crossing error: {response.status_code}"}

@tool
def get_weather(city: str) -> dict:
    """Fetch current weather information from OpenWeatherMap for a specific city."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={w_api_key}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        weather = {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
        return weather
    else:
        return {"error": f"Failed to get weather data: {response.status_code}"}

# Define the state structure
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    city: str
    chart_data: Dict[str, Any]

# Define the system prompt
SYSTEM_PROMPT = """You are a helpful assistant that fetches weather forecasts using different APIs.
You provide weather details in a proper chart and include icons to show weather conditions.
Format time as '21 May 12:00 PM' when presenting results."""

# Define nodes
def call_model(state: AgentState) -> AgentState:
    """Invoke the model with the current messages and tools."""
    messages = state["messages"]
    tools = [get_weatherapi_forecast, get_tomorrow_forecast, get_visualcrossing_weather, get_weather]
    model_with_tools = model.bind_tools(tools)
    
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

def call_tool(state: AgentState) -> AgentState:
    """Execute the tool call specified by the model."""
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return state

    tool_results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool = next((t for t in [get_weatherapi_forecast, get_tomorrow_forecast, get_visualcrossing_weather, get_weather] if t.name == tool_name), None)
        if tool:
            result = tool.invoke(tool_args)
            tool_results.append(result)
            if "city" not in state or not state["city"]:
                state["city"] = tool_args.get("city", "")

    # Prepare chart data if tool call was successful
    chart_data = {}
    if tool_results and "error" not in tool_results[0]:
        result = tool_results[0]
        source = result.get("source", "")
        forecast = result.get("forecast", [])
        if forecast:
            labels = []
            temps = []
            for entry in forecast:
                # Format time for labels
                if source == "Tomorrow.io":
                    dt = datetime.strptime(entry["time"], "%Y-%m-%dT%H:%M:%SZ")
                    labels.append(dt.strftime("%d %b %I:%M %p"))
                    temps.append(entry["temperature"])
                elif source == "Visual Crossing" and "datetime" in entry:
                    dt = datetime.strptime(entry["datetime"], "%Y-%m-%d" if not entry["datetime"].startswith("00:") else "%H:%M:%S")
                    labels.append(dt.strftime("%d %b %I:%M %p") if entry["datetime"].startswith("00:") else dt.strftime("%d %b"))
                    temps.append(entry["temp"])
                elif source == "WeatherAPI":
                    dt = datetime.strptime(entry["date"], "%Y-%m-%d")
                    labels.append(dt.strftime("%d %b"))
                    temps.append(entry["avg_temp"])
                else:  # OpenWeatherMap
                    labels.append(datetime.now().strftime("%d %b %I:%M %p"))
                    temps.append(entry["temperature"])

            chart_data = {
                "labels": labels,
                "temps": temps,
                "source": source
            }

    return {"messages": [AIMessage(content=str(tool_results))], "chart_data": chart_data, "city": state.get("city", "")}

def should_continue(state: AgentState) -> str:
    """Determine if the workflow should continue to tool execution or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "call_tool"
    return END

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("call_model", call_model)
workflow.add_node("call_tool", call_tool)
workflow.add_edge(START, "call_model")
workflow.add_conditional_edges("call_model", should_continue, {"call_tool": "call_tool", END: END})
workflow.add_edge("call_tool", "call_model")
graph = workflow.compile()

# Chainlit handlers
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Welcome to Weather Assistant! Enter your query about weather.").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    history.append(HumanMessage(content=message.content))
    
    # Initialize state
    state = {
        "messages": history,
        "city": "",
        "chart_data": {}
    }
    
    # Execute the graph
    result = await graph.invoke(state)

    # Update history with assistant response
    final_message = result["messages"][-1].content
    history.append(AIMessage(content=final_message))
    cl.user_session.set("history", history)
    
    # Generate chart if applicable
    chart_data = result.get("chart_data", {})
    if chart_data.get("labels") and chart_data.get("temps"):
        chart_config = {
            "type": "line",
            "data": {
                "labels": chart_data["labels"],
                "datasets": [{
                    "label": f"Temperature (°C) - {chart_data['source']}",
                    "data": chart_data["temps"],
                    "borderColor": "#1E90FF",
                    "backgroundColor": "rgba(30, 144, 255, 0.2)",
                    "fill": True
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {"position": "top"},
                    "title": {"display": True, "text": f"Weather Forecast for {result['city']}"}
                },
                "scales": {
                    "y": {"title": {"display": True, "text": "Temperature (°C)"}}
                }
            }
        }
        
        # Send chart
        await cl.Message(content="Here's the weather forecast chart:").send()
        await cl.Message(content="", elements=[cl.Pyplot(chart_config)]).send()
    
    # Send the text response
    await cl.Message(content=final_message).send()