import mesa
import requests
import re

class EconomicAgent(mesa.Agent):
    def __init__(self, unique_id, model, personality, model_name):
        super().__init__(unique_id, model)
        self.wealth = 10
        self.personality = personality
        self.model_name = model_name # This tells the agent which Ollama model to use

    def get_ai_decision(self):
        prompt = f"Personality: {self.personality}. Wealth: {self.wealth}. Pool Multiplier: 1.5x. Contribute 0-5 gold. Response: ONE DIGIT ONLY."
        
        try:
            # We pass self.model_name dynamically to Ollama
            response = requests.post('http://localhost:11434/api/generate', 
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "system": f"Act strictly as a {self.personality}.",
                    "stream": False
                })
            result = response.json()
            answer = result['response'].strip()
            digit = re.search(r'\d', answer)
            return int(digit.group()) if digit else 0
        except:
            return 0

    def step(self):
        contribution = self.get_ai_decision()
        if self.wealth >= contribution:
            self.wealth -= contribution
            self.model.common_pool += contribution
            print(f"Agent {self.unique_id} ({self.model_name}/{self.personality}) gave {contribution}. Wealth: {self.wealth:.2f}")

class PublicGoodsGame(mesa.Model):
    def __init__(self):
        super().__init__()
        self.common_pool = 0
        self.schedule = mesa.time.RandomActivation(self)
        
        # 3 Agents with 3 different brains and 3 different goals
        configs = [
            {"name": "gemma4:e2b", "personality": "Selfless Cooperator"},
            {"name": "llama3.1:8b", "personality": "Rational Accountant"},
            {"name": "dolphin-mistral:7b", "personality": "Greedy Infiltrator"} # Replace with any model you pulled
        ]
        
        for i, config in enumerate(configs):
            a = EconomicAgent(i, self, config['personality'], config['name'])
            self.schedule.add(a)

    def step(self):
        self.common_pool = 0
        self.schedule.step()
        reward = (self.common_pool * 1.5) / 3
        for agent in self.schedule.agents:
            agent.wealth += reward
        print(f"--- Round End | Pool: {self.common_pool} | Reward: {reward:.2f} ---")

if __name__ == "__main__":
    game = PublicGoodsGame()
    for i in range(3):
        print(f"\nROUND {i+1}")
        game.step()
