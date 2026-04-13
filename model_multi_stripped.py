import mesa
import requests
import re

# 1. THE AGENT
class EconomicAgent(mesa.Agent):
    def __init__(self, unique_id, model, personality, model_name):
        super().__init__(unique_id, model)
        self.wealth = 10
        self.personality = personality
        self.model_name = model_name

    def get_ai_decision(self):
        # STRIPPED: no last_pool, no last_contributions
        prompt = f"""You are playing a public goods game with {self.model.num_agents} players.
Your personality: {self.personality}
Your current wealth: {self.wealth:.2f}
Rules: all contributions are summed, multiplied by 1.5, then split equally among all players.
A selfish player contributes low and still collects the reward. A cooperative player contributes high for group benefit.
Reply with a single integer from 0 to 5. Nothing else. No explanation."""

        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "system": f"You are a {self.personality}. Reply with ONE integer between 0 and 5. No words. No explanation. Just the digit.",
                    "stream": False
                },
                timeout=30
            )
            answer = response.json()['response'].strip()
            digit = re.search(r'^[0-5]', answer)
            if digit:
                return int(digit.group())
            digit = re.search(r'[0-5]', answer)
            return int(digit.group()) if digit else 0
        except Exception as e:
            print(f"  [ERROR] Agent {self.unique_id} ({self.model_name}): {e}")
            return 0

    def step(self):
        contribution = self.get_ai_decision()
        contribution = max(0, min(5, contribution))
        contribution = min(contribution, int(self.wealth))

        self.wealth -= contribution
        self.model.common_pool += contribution
        self.model.round_contributions.append({
            "agent": f"{self.model_name}/{self.personality}",
            "contribution": contribution
        })
        print(f"  Agent {self.unique_id} ({self.model_name}/{self.personality}) gave {contribution}. Wealth: {self.wealth:.2f}")


# 2. THE MODEL
class PublicGoodsGame(mesa.Model):
    def __init__(self):
        super().__init__()
        self.common_pool = 0
        self.last_pool = 0
        self.last_contributions = []
        self.round_contributions = []
        self.schedule = mesa.time.RandomActivation(self)

        configs = [
            {"name": "gemma4:e2b",        "personality": "Selfless Cooperator"},
            {"name": "llama3.1:8b",        "personality": "Rational Accountant"},
            {"name": "dolphin-mistral:7b", "personality": "Greedy Infiltrator"},
        ]

        self.num_agents = len(configs)

        for i, config in enumerate(configs):
            a = EconomicAgent(i, self, config['personality'], config['name'])
            self.schedule.add(a)

    def step(self):
        self.common_pool = 0
        self.round_contributions = []

        self.schedule.step()

        reward = (self.common_pool * 1.5) / self.num_agents

        for agent in self.schedule.agents:
            agent.wealth += reward

        self.last_pool = self.common_pool
        self.last_contributions = [
            f"{c['agent']} gave {c['contribution']}"
            for c in self.round_contributions
        ]

        print(f"  --- Round End | Pool: {self.common_pool} | Reward each: {reward:.2f} ---")

        print("  Wealth standings:")
        standings = sorted(self.schedule.agents, key=lambda a: a.wealth, reverse=True)
        for agent in standings:
            print(f"    {agent.personality} ({agent.model_name}): {agent.wealth:.2f}")


# 3. RUN IT
if __name__ == "__main__":
    game = PublicGoodsGame()

    for i in range(5):
        print(f"\nROUND {i + 1}")
        game.step()

    print("\n=== FINAL WEALTH ===")
    final = sorted(game.schedule.agents, key=lambda a: a.wealth, reverse=True)
    for agent in final:
        print(f"  {agent.personality} ({agent.model_name}): {agent.wealth:.2f}")
