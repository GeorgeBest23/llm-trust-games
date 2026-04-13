import mesa

# 1. THE AGENT
class EconomicAgent(mesa.Agent):
    # In Mesa 3.0, we just take the model. unique_id is automatic.
    def __init__(self, model):
        super().__init__(model)
        self.wealth = 10 

    def step(self):
        # Pick a random contribution between 0 and 5
        contribution = self.random.randrange(0, 6)
        
        if self.wealth >= contribution:
            self.wealth -= contribution
            self.model.common_pool += contribution
            print(f"Agent {self.unique_id} contributed {contribution}. Remaining: {self.wealth:.2f}")

# 2. THE MODEL
class PublicGoodsGame(mesa.Model):
    def __init__(self, n_agents):
        # MANDATORY: You must call super().__init__() first in Mesa 3.0
        super().__init__()
        self.num_agents = n_agents
        self.common_pool = 0
        
        # Create agents
        for i in range(self.num_agents):
            # In Mesa 3.0, initializing the agent automatically registers it
            EconomicAgent(self)

    def step(self):
        self.common_pool = 0 
        
        # Tell all agents to act
        self.agents.shuffle_do("step")
        
        # The Multiplier: The "Common Good" grows by 1.5x
        if self.num_agents > 0:
            reward = (self.common_pool * 1.5) / self.num_agents
        else:
            reward = 0
            
        # Redistribute the wealth
        for agent in self.agents:
            agent.wealth += reward
            
        print(f"--- Round End | Pool: {self.common_pool} | Reward per Agent: {reward:.2f} ---")

# 3. RUN THE SIMULATION
if __name__ == "__main__":
    # Create a game with 5 agents
    game = PublicGoodsGame(5)
    
    # Run for 3 rounds
    for i in range(3):
        print(f"\nROUND {i+1}")
        game.step()
