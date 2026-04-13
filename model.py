import mesa

# 1. THE AGENT
class EconomicAgent(mesa.Agent):
    def __init__(self, model):
        # In Mesa 3.0+, we ONLY pass the model. 
        # unique_id is now handled automatically.
        super().__init__(model)
        self.wealth = 10 

    def step(self):
        # The 'heartbeat' - choosing a contribution
        contribution = self.random.randrange(0, 6)
        
        if self.wealth >= contribution:
            self.wealth -= contribution
            # Accessing the model's pool to add money
            self.model.common_pool += contribution
            print(f"Agent {self.unique_id} contributed {contribution}. Wealth: {self.wealth:.2f}")

# 2. THE MODEL
class PublicGoodsGame(mesa.Model):
    def __init__(self, n_agents):
        # Mandatory call to super().__init__() in Mesa 3.0
        super().__init__()
        self.num_agents = n_agents
        self.common_pool = 0
        
        # Create agents - they automatically get added to self.agents
        for i in range(self.num_agents):
            EconomicAgent(self)

    def step(self):
        self.common_pool = 0 
        
        # New way to activate agents in Mesa 3.0: 
        # shuffle_do handles the 'random activation' without a separate scheduler
        self.agents.shuffle_do("step")
        
        # Calculation: Multiply the pool and divide by number of agents
        if self.num_agents > 0:
            reward = (self.common_pool * 1.5) / self.num_agents
        else:
            reward = 0
            
        for agent in self.agents:
            agent.wealth += reward
            
        print(f"--- Round End | Pool: {self.common_pool} | Reward per Agent: {reward:.2f} ---")

# 3. RUN IT
if __name__ == "__main__":
    # Create a 5-agent world
    game = PublicGoodsGame(5)
    
    # Run 3 rounds
    for i in range(3):
        print(f"\nROUND {i+1}")
        game.step()
    
