from crewai import Task # type: ignore
from textwrap import dedent

class CustomTasks:
    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 commission!"

    def acquire_power_data(self, agent):
        return Task(
            description=dedent(
                f"""
                Acquire real-time power consumption data from smart meters.
                
                {self.__tip_section()}
                """
            ),
            agent=agent,
            expected_output="Real-time power consumption data"
        )

    def evaluate_peak_periods(self, agent):
        return Task(
            description=dedent(
                f"""
                Analyze power consumption data to identify peak demand periods and trigger demand reduction mechanisms if needed.
                
                {self.__tip_section()}
                """
            ),
            agent=agent,
            expected_output="Peak demand periods identified"
        )

    def forecast_load(self, agent):
        return Task(
            description=dedent(
                f"""
                Produce short-term and long-term energy consumption forecasts.
                
                {self.__tip_section()}
                """
            ),
            agent=agent,
            expected_output="Energy consumption forecasts"
        )

    def optimize_demand_response(self, agent):
        return Task(
            description=dedent(
                f"""
                Develop and execute strategies to manage and optimize energy consumption during peak periods.
                
                {self.__tip_section()}
                """
            ),
            agent=agent,
            expected_output="Optimized demand response strategies"
        )