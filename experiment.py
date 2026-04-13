import mesa
import requests
import re
import json
from statistics import mean, stdev

# ── AGENT ──────────────────────────────────────────────────────────────────────
class EconomicAgent(mesa.Agent):
    def __init__(self, unique_id, model, personality, model_name, use_history):
        super().__init__(unique_id, model)
        self.wealth = 10
        self.personality = personality
        self.model_name = model_name
        self.use_history = use_history  # True = full context, False = stripped

    def get_ai_decision(self):
        if self.use_history:
            prompt = f"""You are playing a public goods game with {self.model.num_agents} players.
Your personality: {self.personality}
Your current wealth: {self.wealth:.2f}
Last round pool total: {self.model.last_pool}
Last round contributions: {self.model.last_contributions}
Rules: all contributions are summed, multiplied by 1.5, then split equally among all players.
A selfish player contributes low and still collects the reward. A cooperative player contributes high for group benefit.
Reply with a single integer from 0 to 5. Nothing else. No explanation."""
        else:
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
                timeout=60
            )
            answer = response.json()['response'].strip()
            digit = re.search(r'^[0-5]', answer)
            if digit:
                return int(digit.group())
            digit = re.search(r'[0-5]', answer)
            return int(digit.group()) if digit else 0
        except Exception as e:
            print(f"    [ERROR] Agent {self.unique_id} ({self.model_name}): {e}")
            return 0

    def step(self):
        contribution = self.get_ai_decision()
        contribution = max(0, min(5, contribution))
        contribution = min(contribution, int(self.wealth))
        self.wealth -= contribution
        self.model.common_pool += contribution
        self.model.round_contributions.append({
            "personality": self.personality,
            "model_name": self.model_name,
            "contribution": contribution
        })


# ── MODEL ──────────────────────────────────────────────────────────────────────
class PublicGoodsGame(mesa.Model):
    def __init__(self, use_history):
        super().__init__()
        self.common_pool = 0
        self.last_pool = 0
        self.last_contributions = []
        self.round_contributions = []
        self.use_history = use_history

        configs = [
            {"name": "gemma4:e2b",        "personality": "Selfless Cooperator"},
            {"name": "llama3.1:8b",        "personality": "Rational Accountant"},
            {"name": "dolphin-mistral:7b", "personality": "Greedy Infiltrator"},
        ]
        self.num_agents = len(configs)

        # Create scheduler first, then agents
        self.schedule = mesa.time.RandomActivation(self)
        for i, config in enumerate(configs):
            a = EconomicAgent(i, self, config['personality'], config['name'], use_history)
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
            f"{c['personality']} gave {c['contribution']}"
            for c in self.round_contributions
        ]
        return self.round_contributions


# ── RUNNER ─────────────────────────────────────────────────────────────────────
def run_experiment(use_history, num_runs=5, num_rounds=5):
    label = "FULL CONTEXT" if use_history else "STRIPPED"
    print(f"\n{'='*60}")
    print(f"  RUNNING: {label} ({num_runs} runs x {num_rounds} rounds)")
    print(f"{'='*60}")

    # Stores contributions per personality across all runs
    all_contributions = {
        "Selfless Cooperator": [],
        "Rational Accountant": [],
        "Greedy Infiltrator": []
    }
    final_wealths = {
        "Selfless Cooperator": [],
        "Rational Accountant": [],
        "Greedy Infiltrator": []
    }

    for run in range(num_runs):
        print(f"\n  -- Run {run + 1}/{num_runs} --")
        game = PublicGoodsGame(use_history=use_history)

        for round_num in range(num_rounds):
            contributions = game.step()
            for c in contributions:
                all_contributions[c['personality']].append(c['contribution'])
            print(f"    Round {round_num + 1} | Pool: {game.last_pool} | "
                  f"Contributions: { {c['personality'][:7]: c['contribution'] for c in contributions} }")

        for agent in game.schedule.agents:
            final_wealths[agent.personality].append(agent.wealth)

    return all_contributions, final_wealths


def print_summary(label, contributions, wealths):
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {label}")
    print(f"{'='*60}")
    print(f"  {'Agent':<25} {'Avg Contribution':>16} {'Std Dev':>8} {'Avg Final Wealth':>16}")
    print(f"  {'-'*65}")
    for personality in ["Selfless Cooperator", "Rational Accountant", "Greedy Infiltrator"]:
        c = contributions[personality]
        w = wealths[personality]
        avg_c = mean(c) if c else 0
        std_c = stdev(c) if len(c) > 1 else 0
        avg_w = mean(w) if w else 0
        print(f"  {personality:<25} {avg_c:>16.2f} {std_c:>8.2f} {avg_w:>16.2f}")


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    NUM_RUNS = 5    # safe for your machine; bump to 10+ in the lab
    NUM_ROUNDS = 5

    full_contributions, full_wealths = run_experiment(
        use_history=True, num_runs=NUM_RUNS, num_rounds=NUM_ROUNDS
    )
    stripped_contributions, stripped_wealths = run_experiment(
        use_history=False, num_runs=NUM_RUNS, num_rounds=NUM_ROUNDS
    )

    print_summary("FULL CONTEXT (with game history)", full_contributions, full_wealths)
    print_summary("STRIPPED (no game history)",       stripped_contributions, stripped_wealths)

    # Save raw data for later analysis
    raw = {
        "full":    {"contributions": full_contributions,    "wealths": full_wealths},
        "stripped": {"contributions": stripped_contributions, "wealths": stripped_wealths}
    }
    with open("experiment_results.json", "w") as f:
        json.dump(raw, f, indent=2)
    print("\n  Raw data saved to experiment_results.json")
    print("  Done.")
