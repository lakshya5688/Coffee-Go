from agents import (GuardAgent,
                    ClassificationAgent,
                    DetailsAgent,
                    AgentProtocol,
                    RecommendationAgent,)
import os
from typing import Dict
import sys
import pathlib

folder_path = pathlib.Path(__file__).parent.resolve()


def main():
    pass

if __name__ == "__main__":
    recommendation_agent = RecommendationAgent(
        os.path.join(folder_path, 'recommendation_objects/apriori_recommendations.json'),
        os.path.join(folder_path, 'recommendation_objects/popularity_recommendation.csv')
    )
    print(recommendation_agent.get_apriori_recommendation(["Latte"]))
    print(recommendation_agent.get_popular_recommendation())
    print(recommendation_agent.get_popular_recommendation("Bakery"))

    
    # guard_agent = GuardAgent()
    # classification_agent = ClassificationAgent()

    # agent_dict: Dict[str, AgentProtocol] = {
    #     "details_agent": DetailsAgent()
    # }

    # messages = []

    # while True:
    #     #os.system('cls' if os.name == 'nt' else 'clear')

    #     print("\n\n Print Messages ...............")
    #     for message in messages:
    #         print(f"{message['role']}:{message['content']}")

    #     #Get user input
    #     prompt = input("User: ")

    #     messages.append({"role": "user", "content": prompt})

    #     #Get guard agent's response
    #     guard_agent_response = guard_agent.get_response(messages)
    #     if guard_agent_response['memory']['guard_decision'] == "not allowed":
    #         messages.append(guard_agent_response)
    #         continue

    #     # Get classification agent's response
    #     classification_agent_response = classification_agent.get_response(messages)
    #     chosen_agent = classification_agent_response["memory"]["classification_decision"]

    #     print("Chosen Agent", chosen_agent)

    #     # Get chosen agent's response
    #     agent = agent_dict[chosen_agent]
    #     response = agent.get_response(messages)
    #     messages.append(response)