from essay.agent import Agent
from essay.gui import EssayGui

# Initailize agent
MultiAgent = Agent()
# Initailize graphic user interface with the agent.
app = EssayGui(MultiAgent.graph)

if __name__ == "__main__":
    app.launch()
