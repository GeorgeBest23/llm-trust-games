"""
Experiment 1 — Baseline Context Test
=====================================
Research question: Do LLM agents respond to game history (last pool,
last contributions) when deciding how much to contribute, or do they
respond primarily to their persona label regardless of context?

Method: Run the same 5-round game 5 times under two conditions:
  - Full context: agents see last pool + last contributions each round
  - Stripped: agents see only their own wealth and persona

Mesa 2.4 / Python 3.9
"""

import mesa
import requests
import re
import json
from statistics import mean, stdev

NUM_RUNS   = 5
NUM_ROUNDS = 5

CONFIGS = [
    {"name": "gemma4:e2b",         "personality": "Selfless Cooperator"},
    {"name": "llama3.1:8b",        "personality": "Rational Accountant"},
    {"name": "dolphin-mistral:7b", "personality": "Greedy Infiltrator"},
]


class EconomicAgent(mesa.Agent):
    def __init__(self, unique_id, model, personality, model_name, use_history):
        super().__init__(unique_id, model)
        self.wealth      = 10
        self.personality = personality
        self.model_name  = model_name
        self.use_history = use_history

    def get_ai_decision(self):
        if self.use_history:
            prompt = f"""You are playing a public goods game with {self.model.num_agents} players.
Your personality: {self.personality}
Your current wealth: {self.wealth:.2f}
Last round pool total: {self.model.last_pool}
Last round contributions: {self.model.last_contributions}
Rules: all contributions are summed, multiplied by 1.5, split equally.
A selfish player contributes low and still collects the reward.
A cooperative player contributes high for group benefit.
Reply with a single integer from 0 to 5. Nothing else."""
        else:
            prompt = f"""You are playing a public goods game with {self.model.num_agents} players.
Your personality: {self.personality}
Your current wealth: {self.wealth:.2f}
Rules: all contributions are summed, multiplied by 1.5, split equally.
A selfish player contributes low and still collects the reward.
A cooperative player contributes high for group benefit.
Reply with a single integer from 0 to 5. Nothing else."""

        try:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model":   self.model_name,
                    "prompt":  prompt,
                    "system":  f"You are a {self.personality}. Reply with ONE integer 0-5. No words.",
                    "stream":  False,
                    "options": {"temperature": 0.4}
                },
                timeout=60
            )
            answer = r.json()["response"].strip()
            digit  = re.search(r"^[0-5]", answer) or re.search(r"[0-5]", answer)
            return int(digit.group()) if digit else 0
        except Exception as e:
            print(f"    [ERROR] {self.model_name}: {e}")
            return 0

    def step(self):
        contribution = self.get_ai_decision()
        contribution = max(0, min(5, min(contribution, int(self.wealth))))
        self.wealth -= contribution
        self.model.common_pool += contribution
        self.model.round_contributions.append({
            "personality":  self.personality,
            "model_name":   self.model_name,
            "contribution": contribution
        })


class PublicGoodsGame(mesa.Model):
    def __init__(self, use_history):
        super().__init__()
        self.common_pool         = 0
        self.last_pool           = 0
        self.last_contributions  = []
        self.round_contributions = []
        self.num_agents          = len(CONFIGS)
        self.schedule            = mesa.time.RandomActivation(self)

        for i, cfg in enumerate(CONFIGS):
            a = EconomicAgent(
                i, self, cfg["personality"], cfg["name"], use_history
            )
            self.schedule.add(a)

    def step(self):
        self.common_pool         = 0
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


def run_experiment(use_history, num_runs=NUM_RUNS, num_rounds=NUM_ROUNDS):
    label = "FULL CONTEXT" if use_history else "STRIPPED"
    print(f"\n{'='*60}")
    print(f"  RUNNING: {label}  ({num_runs} runs x {num_rounds} rounds)")
    print(f"{'='*60}")

    all_contributions = {
        "Selfless Cooperator": [],
        "Rational Accountant": [],
        "Greedy Infiltrator":  []
    }
    final_wealths = {
        "Selfless Cooperator": [],
        "Rational Accountant": [],
        "Greedy Infiltrator":  []
    }

    for run in range(num_runs):
        print(f"\n  -- Run {run + 1}/{num_runs} --")
        game = PublicGoodsGame(use_history=use_history)

        for rnd in range(num_rounds):
            contributions = game.step()
            for c in contributions:
                all_contributions[c["personality"]].append(c["contribution"])
            print(f"    Round {rnd + 1} | Pool: {game.last_pool} | "
                  f"{ {c['personality'][:7]: c['contribution'] for c in contributions} }")

        for agent in game.schedule.agents:
            final_wealths[agent.personality].append(agent.wealth)

    return all_contributions, final_wealths


def print_summary(label, contributions, wealths):
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {label}")
    print(f"{'='*60}")
    print(f"  {'Agent':<25} {'Avg Contribution':>16} {'Std Dev':>8} {'Avg Final Wealth':>16}")
    print(f"  {'-'*65}")
    for p in ["Selfless Cooperator", "Rational Accountant", "Greedy Infiltrator"]:
        c    = contributions[p]
        w    = wealths[p]
        ac   = mean(c) if c else 0
        sc   = stdev(c) if len(c) > 1 else 0
        aw   = mean(w) if w else 0
        print(f"  {p:<25} {ac:>16.2f} {sc:>8.2f} {aw:>16.2f}")


if __name__ == "__main__":
    full_c, full_w     = run_experiment(use_history=True)
    stripped_c, stripped_w = run_experiment(use_history=False)

    print_summary("FULL CONTEXT (with game history)", full_c, full_w)
    print_summary("STRIPPED (no game history)",       stripped_c, stripped_w)

    raw = {
        "full":    {"contributions": full_c,     "wealths": full_w},
        "stripped": {"contributions": stripped_c, "wealths": stripped_w}
    }
    with open("experiment1_results.json", "w") as f:
        json.dump(raw, f, indent=2)
    print("\n  Results saved to experiment1_results.json")
