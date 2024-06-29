from crewai import Agent
from textwrap import dedent
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI

class CustomAgents:
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4", temperature=0.7)

    def power_data_monitor(self):
        return Agent(
            role="Power Data Monitor",
            backstory=dedent("""This agent monitors real-time power consumption data from smart meters."""),
            goal=dedent("""Collect and provide real-time power consumption data, including voltage and frequency."""),
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def peak_period_evaluator(self):
        return Agent(
            role="Peak Period Evaluator",
            backstory=dedent("""This agent evaluates power consumption data to identify peak demand periods."""),
            goal=dedent("""Analyze data to determine if the grid is within peak demand periods and trigger demand reduction mechanisms."""),
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def load_forecasting_agent(self):
        return Agent(
            role="Load Forecasting Agent",
            backstory=dedent("""This agent predicts future energy consumption using historical data, weather conditions, and other relevant factors."""),
            goal=dedent("""Provide accurate and reliable energy consumption forecasts to aid in planning and decision-making."""),
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4,
        )

    def demand_response_optimizer_agent(self):
        return Agent(
            role="Demand Response Optimizer Agent",
            backstory=dedent("""This agent develops and executes strategies to manage and optimize energy consumption during peak periods."""),
            goal=dedent("""Optimize demand response actions by determining the best strategies to reduce or shift energy consumption."""),
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4,
        )