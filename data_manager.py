import streamlit as st
import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain_core.runnables import RunnableSequence
from langchain.agents import Tool,initialize_agent


# Initialize OpenAI API
load_dotenv(find_dotenv())
openai_api_key=os.getenv("OPENAI_API_KEY")
openai_llm = OpenAI(temperature=0, openai_api_key=openai_api_key)


# Define prompt templates
exercise_prompt = PromptTemplate(
    input_variables=["hours", "type"],
    template="I exercised for {hours} hours doing {type}. How can I improve?"
)

diet_prompt = PromptTemplate(
    input_variables=["diet"],
    template="I followed this diet plan: {diet}. How can I improve?"
)

water_prompt = PromptTemplate(
    input_variables=["water"],
    template="I drank {water} liters of water. How can I improve?"
)

sleep_prompt = PromptTemplate(
    input_variables=["sleep"],
    template="I slept for {sleep} hours. How can I improve?"
)
llm_chain = RunnableSequence(diet_prompt | openai_llm)
#initialize the tool
llm_tool = Tool(
    name="Dietician Model",
    func=llm_chain.invoke,
    description="This tool acts as a dietician and provides suggestions on how to improve one's diet plan for a borderline diabetic patient based on the diet plan they give."
)
tools=[llm_tool]
#initialize the agent
agent=initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=openai_llm,
    verbose=True,
    max_iterations=3 #number of iterations to run the agent,we limit it to 3
)

# Streamlit app
st.title("Health Assistant App")
# Input fields
exercise_hours = st.number_input("Hours spent exercising", min_value=0.0, step=0.5)
exercise_type = st.selectbox("Type of exercise", ["None","Walking", "Jogging", "Running", "Dancing", "Yoga"])
diet_plan = st.text_area("What did you eat today?")
water_intake = st.number_input("Amount of water drunk (liters)", min_value=0.0, step=0.1)
sleep_hours = st.number_input("Sleep hours", min_value=0.0, step=0.5)

# Generate suggestions
if st.button("Get Suggestions"):
    try:
        exercise_suggestion = openai_llm(exercise_prompt.format(hours=exercise_hours, type=exercise_type))
        diet_suggestion = agent.invoke(diet_plan)
        water_suggestion = openai_llm(water_prompt.format(water=water_intake))
        sleep_suggestion = openai_llm(sleep_prompt.format(sleep=sleep_hours))

        
        # Extract the output part from the diet suggestion
        diet_output = diet_suggestion.get("output", "No suggestion available")
        diet_output = diet_output.replace("Based on the Dietician Model, the final answer is", "As a dietician, I suggest you make these changes to your diet:")
        # Display suggestions
        st.subheader("Suggestions for Improvement")
        st.write("Exercise: ", exercise_suggestion)
        st.write("Diet: ", diet_output)
        st.write("Water Intake: ", water_suggestion)
        st.write("Sleep: ", sleep_suggestion)

        st.subheader("To summarize:")
        if exercise_hours >= 1:
            st.write("Great job on exercising!")
        else:
            st.write("Exercise: You need to really work on exercising.It is a very crucial part of your health.")
        if water_intake < 1:
            st.write("Water Intake: Hydration is very essential. Try to drink more water.")
        elif 1 <= water_intake < 2:
            st.write("Water Intake: Keep it up and try drinking a bit more.")
        else:
            st.write("Water Intake: Good job!")
        if sleep_hours >= 7:
            st.write("Well done on getting enough sleep!")
        else:
            st.write("Sleep: You need to work on getting more sleep.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
st.write("Everyone finds it hard to resist unhealthy food and maintain a diet. Watch the video below to learn more about how to improve your diet.")
st.video("https://www.youtube.com/watch?v=O9ouhTy2QBU")