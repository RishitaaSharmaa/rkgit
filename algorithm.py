import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

api_key=os.getenv("GROQ_API")

llm = ChatGroq(api_key=api_key,model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

template = """
You are an expert in Rainwater Harvesting Systems and Artificial Recharge.
Inputs:
- Location: {location}
- Roof type: {roof_type}
- Roof material: {roof_material}
- Number of dwellers: {dwellers}
- Soil type: {soil_type}
- Groundwater level: {groundwater_level}
- Roof area (sq.m): {roof_area}
- Current harvested water: {current_volume} L
- Household size: {dwellers}
- Avg daily demand/person: 5 L
- Forecasted rainfall: {rainfall} mm
- Roof area: {roof_area} m²
- Water tariff: {tariff} ₹/KL

Task:
- Feasibility check for rooftop rainwater harvesting.
- Suggested type of RTRWH and Artificial Recharge structures.
- Runoff generation capacity.
- Recommended dimensions of recharge pits, trenches, and shafts.
- Cost estimation and cost-benefit analysis.
- Estimate how many days this water will last.
- Predict future harvest from rainfall.
- Calculate money saved.
- Calculate daily water demand for the household.
- Explain impact in user-friendly language.

"""

prompt = PromptTemplate(
    input_variables=["location", "roof_type", "roof_material", "dwellers", 
                     "soil_type", "groundwater_level", "roof_area","current_volume", "dwellers", "rainfall", "roof_area","tariff"],
    template=template
)

chain = prompt|llm

user_inputs = {
    "location": "Jaipur, Rajasthan",
    "roof_type": "Flat",
    "roof_material": "Concrete",
    "dwellers": "4",
    "soil_type": "Sandy",
    "groundwater_level": "25 meters",
    "roof_area": "120",
    "current_volume": 200,
    "dwellers": 12,
    "rainfall": 8,
    "roof_area": 15,   
    "tariff": 15
}

response = chain.invoke(user_inputs)
print("Recommendation:\n", response.content)
