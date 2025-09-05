import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

api_key=os.getenv("GROQ_API")

llm = ChatGroq(api_key=api_key,model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

template = """
You are a water resource advisor.
Inputs:
- Current harvested water: {current_volume} L
- Household size: {dwellers}
- Avg daily demand/person: 135 L
- Forecasted rainfall: {rainfall} mm
- Roof area: {roof_area} m²
- Water tariff: {tariff} ₹/KL

Task:
1. Estimate how many days this water will last.
2. Predict future harvest from rainfall.
3. Calculate money saved.
4. Calculate daily water demand for the household.
5. Explain impact in user-friendly language.

Provide the answer in short actionable insights.
"""

prompt = PromptTemplate(
    input_variables=[
        "current_volume", "dwellers", "rainfall", "roof_area",
        "tariff"
    ],
    template=template
)

chain = prompt|llm

user_inputs = {
    "current_volume": 200,
    "dwellers": 12,
    "rainfall": 8,
    "roof_area": 15,   
    "tariff": 15
}

response = chain.invoke(user_inputs)
print("Recommendation:\n", response.content)
