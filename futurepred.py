import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

api_key=os.getenv("GROQ_API")

llm = ChatGroq(api_key=api_key,model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

template = """
You are a water sustainability expert. 
Using the provided inputs, calculate and explain the household's water sustainability projection.

Inputs:
- Current stored water: {current_storage} litres
- Roof area: {roof_area} m²
- Runoff coefficient: {runoff_coeff}
- Forecasted rainfall (mm over next 7 days): {rainfall_forecast}
- Number of dwellers: {dwellers}
- Per capita demand: {lpcd} litres/day
- Water tariff: ₹{tariff}/kL

Task:
1. Estimate additional water that can be collected from forecasted rainfall.
2. Calculate daily water demand for the household.
3. Estimate how many days the total water will last.
4. Estimate total water savings and monetary savings.
5. Explain the environmental impact in simple terms.
6. Present results clearly in bullet points with numbers.
"""

prompt = PromptTemplate(
    input_variables=[
        "current_storage", "roof_area", "runoff_coeff", "rainfall_forecast",
        "dwellers", "lpcd", "tariff"
    ],
    template=template
)


chain = prompt|llm

user_inputs = {
    "current_storage": 2000,
    "roof_area": 120,
    "runoff_coeff": 0.8,
    "rainfall_forecast": 15,   
    "dwellers": 6,
    "lpcd": 135,
    "tariff": 15
}


response = chain.invoke(user_inputs)
print("Recommendation:\n", response.content)
