import mesa

# 1. THE AGENT
class EconomicAgent(mesa.Agent):
    def __init__(self, unique_id, model):          # unique_id required in 2.x
        super().__init__(unique_id, model)          # must pass both
        self.wealth = 10

    def step(self):
        contribution = self.random.randrange(0, 6)
        if self.wealth >= contribution:
            self.wealth -= contribution
            self.model.common_pool += contribution
            print(f"Agent {self.unique_id} contributed {contribution}. Remaining: {self.wealth:.2f}")

# 2. THE MODEL
class PublicGoodsGame(mesa.Model):
    def __init__(self, n_agents):
        super().__init__()
        self.num_agents = n_agents
        self.common_pool = 0
        self.schedule = mesa.time.RandomActivation(self)

        for i in range(self.num_agents):
            a = EconomicAgent(i, self)             # pass i as unique_id
            self.schedule.add(a)

    def step(self):
        self.common_pool = 0
        self.schedule.step()

        reward = (self.common_pool * 1.5) / self.num_agents if self.num_agents > 0 else 0

        for agent in self.schedule.agents:
            agent.wealth += reward

        print(f"--- Round End | Pool: {self.common_pool} | Reward per Agent: {reward:.2f} ---")

# 3. RUN IT
if __name__ == "__main__":
    game = PublicGoodsGame(5)
    for i in range(3):
        print(f"\nROUND {i+1}")
        game.step()
