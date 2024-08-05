import os
import base64
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from textwrap import dedent
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import openai

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai

# Function to convert text to audio
def text_to_audio(client, text, audio_path):
    audio_text = str(text)
    response = client.audio.speech.create(model="tts-1", voice="fable", input=audio_text)
    response.stream_to_file(audio_path)

# Function to autoplay audio
def autoplay_audio(audio_file):
    with open(audio_file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    audio_html = (
        f'<audio src="data:audio/mp3;base64,{base64_audio}" controls autoplay></audio>'
    )
    st.sidebar.markdown(audio_html, unsafe_allow_html=True)

class CustomAgents:
    def power_data_monitor(self, threshold_consumption, threshold_voltage, threshold_frequency):
        # Implementation of the agent that monitors power data
        # and uses the threshold parameters
        return Agent(
            name="Power Data Monitor",
            role="Monitor power data and analyze it",
            goal="Identify any anomalies or issues in the power data",
            backstory="I am an expert in power systems and have extensive experience in analyzing power data.",
            threshold_consumption=threshold_consumption,
            threshold_voltage=threshold_voltage,
            threshold_frequency=threshold_frequency
        )

    def peak_period_evaluator(self):
        # Implementation of the peak period evaluator agent
        return Agent(
            name="Peak Period Evaluator",
            role="Analyze power consumption data to identify peak periods",
            goal="Provide insights into the peak periods and their impact on the power grid",
            backstory="I am an expert in energy analysis and have experience in identifying peak periods in power consumption data."
        )

    def load_forecasting_agent(self):
        # Implementation of the load forecasting agent
        return Agent(
            name="Load Forecasting Agent",
            role="Forecast power consumption based on historical data",
            goal="Provide accurate forecasts to optimize power grid operations",
            backstory="I am an expert in time series forecasting and have experience in forecasting power consumption."
        )

    def demand_response_optimizer_agent(self):
        # Implementation of the demand response optimizer agent
        return Agent(
            name="Demand Response Optimizer",
            role="Optimize demand response strategies based on forecasted load and grid conditions",
            goal="Maximize the efficiency and effectiveness of demand response programs",
            backstory="I am an expert in demand response optimization and have experience in developing strategies to manage power consumption."
        )

class CustomTasks:
    def acquire_power_data(self, power_data_monitor):
        return Task(
            name="Acquire Power Data Task",
            description="The task is to acquire power data from the data source",
            expected_output="A dataframe containing the acquired power data",
            agent=power_data_monitor,
            action="Acquiring power data"
        )

    def evaluate_peak_periods(self, peak_period_evaluator):
        return Task(
            name="Evaluate Peak Periods Task",
            description="The task is to analyze the power consumption data to identify peak periods",
            expected_output="A report containing insights into the peak periods and their impact on the power grid",
            agent=peak_period_evaluator,
            action="Evaluating peak periods"
        )

    def forecast_load(self, load_forecasting_agent):
        return Task(
            name="Forecast Load Task",
            description="The task is to forecast power consumption based on historical data",
            expected_output="A dataframe containing the forecasted power consumption",
            agent=load_forecasting_agent,
            action="Forecasting load"
        )

    def optimize_demand_response(self, demand_response_optimizer_agent):
        return Task(
            name="Optimize Demand Response Task",
            description="The task is to optimize demand response strategies based on forecasted load and grid conditions",
            expected_output="A report containing the optimized demand response strategies and their impact on power grid operations",
            agent=demand_response_optimizer_agent,
            action="Optimizing demand response"
        )

class CustomCrew:
    def __init__(self, threshold_consumption, threshold_voltage, threshold_frequency):
        self.agents = CustomAgents()
        self.tasks = CustomTasks()
        self.threshold_consumption = threshold_consumption
        self.threshold_voltage = threshold_voltage
        self.threshold_frequency = threshold_frequency

    def run(self):
        # Define agents with the necessary parameters
        power_data_monitor = self.agents.power_data_monitor(self.threshold_consumption, self.threshold_voltage, self.threshold_frequency)
        peak_period_evaluator = self.agents.peak_period_evaluator()
        load_forecasting_agent = self.agents.load_forecasting_agent()
        demand_response_optimizer_agent = self.agents.demand_response_optimizer_agent()

        # Define tasks
        acquire_power_data_task = self.tasks.acquire_power_data(power_data_monitor)
        evaluate_peak_periods_task = self.tasks.evaluate_peak_periods(peak_period_evaluator)
        forecast_load_task = self.tasks.forecast_load(load_forecasting_agent)
        optimize_demand_response_task = self.tasks.optimize_demand_response(demand_response_optimizer_agent)

        # Define crew
        crew = Crew(
            agents=[
                power_data_monitor,
                peak_period_evaluator,
                load_forecasting_agent,
                demand_response_optimizer_agent
            ],
            tasks=[
                acquire_power_data_task,
                evaluate_peak_periods_task,
                forecast_load_task,
                optimize_demand_response_task
            ],
            verbose=True,
        )

        try:
            result = crew.kickoff()
        except Exception as e:
            result = f"An error occurred: {str(e)}"
        return result

# Streamlit App
def main():
    st.title("Power Consumption Monitoring Application")

    st.sidebar.header("Input Parameters")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")

    st.sidebar.header("Agent Parameters")
    threshold_consumption = st.sidebar.number_input("Threshold Consumption (kWh)", min_value=0, value=500)
    threshold_voltage = st.sidebar.number_input("Threshold Voltage (V)", min_value=0, value=240)
    threshold_frequency = st.sidebar.number_input("Threshold Frequency (Hz)", min_value=0, value=50)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Clean column headers
        df.columns = df.columns.str.replace(' ', '').str.lower()

        # Check if the required columns exist in the uploaded CSV file
        required_columns = ['time', 'power', 'voltage', 'frequency']
        if all(column in df.columns for column in required_columns):
            custom_crew = CustomCrew(threshold_consumption, threshold_voltage, threshold_frequency)
            result = custom_crew.run()

            st.write("## Real-Time Power Consumption Data")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['time'], y=df['power'], mode='lines+markers', name='Power Consumption'))
            fig.add_trace(go.Scatter(x=df['time'], y=df['voltage'], mode='lines+markers', name='Voltage'))
            fig.add_trace(go.Scatter(x=df['time'], y=df['frequency'], mode='lines+markers', name='Frequency'))

            fig.update_layout(
                title="Power Consumption Data",
                xaxis_title="Time",
                yaxis_title="Measurements",
                legend_title="Legend",
                xaxis=dict(rangeslider=dict(visible=True))  # Add scroll slider
            )

            st.plotly_chart(fig)

            st.write("## Load Forecasting Data")
            df_forecast = pd.DataFrame({
                'Time': pd.date_range(start='1/1/2024', periods=24, freq='h'),
                'Forecasted Consumption': np.random.randint(100, 1000, size=24)
            })

            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=df_forecast['Time'], y=df_forecast['Forecasted Consumption'], mode='lines+markers', name='Forecasted Consumption'))

            fig_forecast.update_layout(
                title="Forecasted Power Consumption",
                xaxis_title="Time",
                yaxis_title="Forecasted Consumption (kWh)",
                legend_title="Legend"
            )

            st.plotly_chart(fig_forecast)

            st.write("## Demand Response Optimization Impact")
            df_optimization = pd.DataFrame({
                'Time': pd.date_range(start='1/1/2024', periods=24, freq='h'),
                'Optimized Load': np.random.randint(100, 1000, size=24)
            })

            fig_optimization = go.Figure()
            fig_optimization.add_trace(go.Scatter(x=df_optimization['Time'], y=df_optimization['Optimized Load'], mode='lines+markers', name='Optimized Load'))

            fig_optimization.update_layout(
                title="Optimized Load After Demand Response",
                xaxis_title="Time",
                yaxis_title="Optimized Load (kWh)",
                legend_title="Legend"
            )

            st.plotly_chart(fig_optimization)

            st.write("## Agents' Output")
            st.write(result)

            if isinstance(result, str):
                st.error(result)
            else:
                # Convert agents' output to audio and display it on the sidebar
                audio_text = result
                audio_path = "output.mp3"
                text_to_audio(client, audio_text, audio_path)
                autoplay_audio(audio_path)
        else:
            st.error("The uploaded CSV file does not contain the required columns: 'time', 'power', 'voltage', and 'frequency'.")
    else:
        st.info("Please upload a CSV file.")

if __name__ == "__main__":
    main()


