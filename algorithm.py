import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

api_key=os.getenv("GROQ_API")

llm = ChatGroq(api_key=api_key,model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

template = """
You are an expert in Rainwater Harvesting Systems (RRWHS) and Artificial Recharge (AR).
Based on the user's inputs, recommend the most suitable system.

Inputs:
- Location: {location}
- Roof type: {roof_type}
- Roof material: {roof_material}
- Number of dwellers: {dwellers}
- Soil type: {soil_type}
- Groundwater level: {groundwater_level}
- Roof area (sq.m): {roof_area}

Task:
Recommend a single system which is best for the user fufilling the use of both rooftop rainwater harvesting system and aritficial recharge.
And the Estimate cost range. 
Just give the relevant answer and no additional information.
"""

prompt = PromptTemplate(
    input_variables=["location", "roof_type", "roof_material", "dwellers", 
                     "soil_type", "groundwater_level", "roof_area"],
    template=template
)

chain = prompt|llm

user_inputs = {
    "location": "Jaipur, Rajasthan",
    "roof_type": "Flat",
    "roof_material": "Concrete",
    "dwellers": "6",
    "soil_type": "Sandy",
    "groundwater_level": "25 meters",
    "roof_area": "120"
}

response = chain.invoke(user_inputs)
print("💧 Recommendation:\n", response.content)
