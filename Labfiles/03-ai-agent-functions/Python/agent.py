import os
from dotenv import load_dotenv
from typing import Any
from pathlib import Path

# Azure imports
from azure.identity import DefaultAzureCredential
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    FunctionTool,
    ToolSet,
    ListSortOrder,
    MessageRole
)

from user_functions import user_functions


def main():

    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    # Load environment variables
    load_dotenv()
    project_endpoint = os.getenv("PROJECT_ENDPOINT")
    model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

    # Connect to the Agent client
    agent_client = AgentsClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True
        )
    )

    # Define agent with function tools
    with agent_client:

        # Setup function tools
        functions = FunctionTool(user_functions)
        toolset = ToolSet()
        toolset.add(functions)

        # Enable automatic function calling
        agent_client.enable_auto_function_calls(toolset)

        # Create agent
        agent = agent_client.create_agent(
            model=model_deployment,
            name="support-agent",
            instructions=(
                "You are a technical support agent.\n"
                "When a user has a technical issue, you collect their email and problem description.\n"
                "Then call the function to submit a support ticket.\n"
                "If a file is saved, tell the user the file name."
            ),
            toolset=toolset
        )

        # Create thread
        thread = agent_client.threads.create()
        print(f"You're chatting with: {agent.name} ({agent.id})\n")

        # Conversation loop
        while True:
            user_prompt = input("Enter a prompt (or type 'quit' to exit): ")

            if user_prompt.lower() == "quit":
                break

            if not user_prompt.strip():
                print("Please enter a prompt.")
                continue

            # Send user message
            message = agent_client.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_prompt
            )

            # Run agent
            run = agent_client.runs.create_and_process(
                thread_id=thread.id,
                agent_id=agent.id
            )

            if run.status == "failed":
                print(f"Run failed: {run.last_error}")
                continue

            # Display agent response
            last_msg = agent_client.messages.get_last_message_text_by_role(
                thread_id=thread.id,
                role=MessageRole.AGENT
            )

            if last_msg:
                print("\nAgent:", last_msg.text.value, "\n")

        # Print conversation history
        print("\nConversation Log:\n")
        messages = agent_client.messages.list(
            thread_id=thread.id,
            order=ListSortOrder.ASCENDING
        )

        for message in messages:
            if message.text_messages:
                last_msg = message.text_messages[-1]
                print(f"{message.role}: {last_msg.text.value}\n")

        # Cleanup
        agent_client.delete_agent(agent.id)
        print("Deleted agent")


if __name__ == '__main__':
    main()
