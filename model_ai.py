import mesa
import requests
import json

class EconomicAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = 10

    def get_ai_decision(self):
        # The Prompt: Tell the AI who it is and what the stakes are
        prompt = f"""
        You are a player in a Public Goods Game. 
        Your current wealth is {self.wealth}.
        You can contribute between 0 and 5 gold to a common pool.
        The total pool will be multiplied by 1.5 and split equally.
        How much do you contribute? 
        Respond with ONLY a single number from 0 to 5. No text.
        """
        
        try:
            response = requests.post('http://localhost:11434/api/generate', 
                json={
                    "model": "gemma4:e2b", # Ensure you have run 'ollama pull gemma4:e2b'
                    "prompt": prompt,
                    "stream": False
                })
            result = response.json()
            # Try to grab just the first digit it gives us
            answer = result['response'].strip()
            return int(answer[0]) 
        except Exception as e:
            # If the AI fails or talks too much, fall back to 0
            return 0

    def step(self):
        contribution = self.get_ai_decision()
        
        if self.wealth >= contribution:
            self.wealth -= contribution
            self.model.common_pool += contribution
            print(f"Agent {self.unique_id} (AI) contributed {contribution}. Remaining: {self.wealth:.2f}")

# (The PublicGoodsGame class stays exactly the same as your working version)
class PublicGoodsGame(mesa.Model):
    def __init__(self, n_agents):
        super().__init__()
        self.num_agents = n_agents
        self.common_pool = 0
        self.schedule = mesa.time.RandomActivation(self)
        for i in range(self.num_agents):
            a = EconomicAgent(i, self)
            self.schedule.add(a)

    def step(self):
        self.common_pool = 0
        self.schedule.step()
        reward = (self.common_pool * 1.5) / self.num_agents if self.num_agents > 0 else 0
        for agent in self.schedule.agents:
            agent.wealth += reward
        print(f"--- Round End | Pool: {self.common_pool} | Reward: {reward:.2f} ---")

if __name__ == "__main__":
    game = PublicGoodsGame(3) # Let's start with 3 agents to keep it fast
    for i in range(2): # Just 2 rounds for the first test
        print(f"\nROUND {i+1}")
        game.step()
