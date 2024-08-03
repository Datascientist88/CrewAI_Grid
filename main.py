import os
import base64
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from textwrap import dedent
from agents import CustomAgents
from tasks import CustomTasks
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
    response = client.audio.speech.create(model="tts-1", voice="fable", input=text)
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

class CustomCrew:
    def __init__(self):
        self.agents = CustomAgents()
        self.tasks = CustomTasks()

    def run(self):
        # Define agents
        power_data_monitor = self.agents.power_data_monitor()
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

        result = crew.kickoff()
        return result

# Streamlit App
def main():
    st.title("Power Consumption Monitoring Application")

    st.sidebar.header("Input Parameters")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Check if the required columns exist in the uploaded CSV file
        required_columns = ['Time', 'Power', 'Voltage', 'Frequency']
        if all(column in df.columns for column in required_columns):
            custom_crew = CustomCrew()
            result = custom_crew.run()

            st.write("## Real-Time Power Consumption Data")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Time'], y=df['Power'], mode='lines+markers', name='Power Consumption'))
            fig.add_trace(go.Scatter(x=df['Time'], y=df['Voltage'], mode='lines+markers', name='Voltage'))
            fig.add_trace(go.Scatter(x=df['Time'], y=df['Frequency'], mode='lines+markers', name='Frequency'))

            fig.update_layout(
                title="Power Consumption Data",
                xaxis_title="Time",
                yaxis_title="Measurements",
                legend_title="Legend"
            )

            st.plotly_chart(fig)

            st.write("## Load Forecasting Data")
            df_forecast = pd.DataFrame({
                'Time': pd.date_range(start='1/1/2024', periods=24, freq='H'),
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
                'Time': pd.date_range(start='1/1/2024', periods=24, freq='H'),
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

            # Convert agents' output to audio and display it on the sidebar
            audio_text = result
            audio_path = "output.mp3"
            text_to_audio(client, audio_text, audio_path)
            autoplay_audio(audio_path)
        else:
            st.error("The uploaded CSV file does not contain the required columns: 'Time', 'Power', 'Voltage', and 'Frequency'.")
    else:
        st.info("Please upload a CSV file.")

if __name__ == "__main__":
    main()
