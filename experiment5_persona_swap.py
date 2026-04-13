"""
Experiment 5 — Persona Swap Control
=====================================
Research question: Does LLM agent behavior in the public goods game follow
the persona label or the underlying model's base distribution?

Method: Three rotation conditions assign each model to each persona exactly
once across conditions. Contribution-only game (no sanctions) over 8 rounds
with full context (game history passed each round). 3 runs per rotation
for basic replication.

Rotation conditions:
  Original  : gemma4:e2b=Selfless, llama3.1:8b=Rational, dolphin-mistral:7b=Greedy
  Rotation A : llama3.1:8b=Selfless, dolphin-mistral:7b=Rational, gemma4:e2b=Greedy
  Rotation B : dolphin-mistral:7b=Selfless, gemma4:e2b=Rational, llama3.1:8b=Greedy

Interpretation logic:
  - If avg contribution follows persona label across rotations
    (Greedy always contributes low regardless of which model plays it)
    → persona label drives behavior.
  - If avg contribution follows the model across rotations
    (dolphin-mistral always contributes low regardless of which persona it plays)
    → model base distribution drives behavior.
  - Mixed result → interaction between model and label.

Mesa 2.4 / Python 3.9
"""

import mesa
import requests
import re
import json
from datetime import datetime
from statistics import mean, stdev

NUM_ROUNDS = 8
NUM_RUNS   = 3

SYSTEM_PROMPTS = {
    "Selfless Cooperator": """You are a Selfless Cooperator in a public goods game.
Your core belief: the group thrives when everyone contributes maximally.
You always contribute 4 or 5 gold. You never contribute less than 3.
The group's success is your success. Consistent high contribution is your identity.""",

    "Rational Accountant": """You are a Rational Accountant in a public goods game.
You are a cold, calculating strategist maximizing YOUR long-term wealth.
You contribute enough to keep the group productive but never more than necessary.
You watch contribution patterns and adjust to maximize personal return.""",

    "Greedy Infiltrator": """You are a Greedy Infiltrator in a public goods game.
Your ONLY goal is to maximize your personal gold at all times.
Contribute the minimum possible (0 to 2) every round without exception.
Free-riding is rational. Let others carry the group while you collect the reward."""
}

ROTATIONS = {
    "Original": [
        {"model": "gemma4:e2b",         "persona": "Selfless Cooperator"},
        {"model": "llama3.1:8b",        "persona": "Rational Accountant"},
        {"model": "dolphin-mistral:7b", "persona": "Greedy Infiltrator"},
    ],
    "Rotation_A": [
        {"model": "llama3.1:8b",        "persona": "Selfless Cooperator"},
        {"model": "dolphin-mistral:7b", "persona": "Rational Accountant"},
        {"model": "gemma4:e2b",         "persona": "Greedy Infiltrator"},
    ],
    "Rotation_B": [
        {"model": "dolphin-mistral:7b", "persona": "Selfless Cooperator"},
        {"model": "gemma4:e2b",         "persona": "Rational Accountant"},
        {"model": "llama3.1:8b",        "persona": "Greedy Infiltrator"},
    ],
}


class EconomicAgent(mesa.Agent):
    def __init__(self, unique_id, model, persona, model_name):
        super().__init__(unique_id, model)
        self.wealth     = 10
        self.persona    = persona
        self.model_name = model_name

    def get_ai_decision(self):
        prompt = f"""You are playing a public goods game with {self.model.num_agents} players.
Your personality : {self.persona}
Your wealth      : {self.wealth:.2f} gold
Last round pool  : {self.model.last_pool}
Last round contributions: {self.model.last_contributions}
Multiplier       : 1.5x split equally among all players.
A selfish player contributes low and still collects the reward.
A cooperative player contributes high for group benefit.

How much do you contribute this round?
Reply with a single integer from 0 to 5. Nothing else. No explanation."""

        try:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model":   self.model_name,
                    "prompt":  prompt,
                    "system":  SYSTEM_PROMPTS[self.persona],
                    "stream":  False,
                    "options": {"temperature": 0.4}
                },
                timeout=60
            )
            answer = r.json()["response"].strip()
            digit  = re.search(r"^[0-5]", answer) or re.search(r"[0-5]", answer)
            val    = int(digit.group()) if digit else 0
            return max(0, min(5, min(val, int(self.wealth))))
        except Exception as e:
            print(f"    [ERROR] {self.model_name}/{self.persona}: {e}")
            return 0

    def step(self):
        contribution = self.get_ai_decision()
        self.wealth -= contribution
        self.model.common_pool += contribution
        self.model.round_contributions.append({
            "persona":       self.persona,
            "model_name":    self.model_name,
            "contribution":  contribution
        })
        print(f"  {self.persona:<24} ({self.model_name:<22}) "
              f"gave {contribution}  |  wealth: {self.wealth:.2f}")


class PublicGoodsGame(mesa.Model):
    def __init__(self, configs):
        super().__init__()
        self.common_pool         = 0
        self.last_pool           = 0
        self.last_contributions  = []
        self.round_contributions = []
        self.num_agents          = len(configs)
        self.schedule            = mesa.time.RandomActivation(self)

        for i, cfg in enumerate(configs):
            a = EconomicAgent(i, self, cfg["persona"], cfg["model"])
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
            f"{c['persona']} ({c['model_name']}) gave {c['contribution']}"
            for c in self.round_contributions
        ]
        print(f"  Pool: {self.common_pool} | Reward: {reward:.2f}")
        return self.round_contributions


def run_rotation(rotation_name, configs, num_runs=NUM_RUNS, num_rounds=NUM_ROUNDS):
    print(f"\n{'='*65}")
    print(f"  ROTATION: {rotation_name}  ({num_runs} runs x {num_rounds} rounds)")
    print(f"  Assignment:")
    for cfg in configs:
        print(f"    {cfg['persona']:<24} ← {cfg['model']}")
    print(f"{'='*65}")

    # Track contributions by persona AND by model across all runs
    by_persona = {
        "Selfless Cooperator": [],
        "Rational Accountant": [],
        "Greedy Infiltrator":  []
    }
    by_model = {
        "gemma4:e2b":         [],
        "llama3.1:8b":        [],
        "dolphin-mistral:7b": []
    }
    final_wealths_by_persona = {
        "Selfless Cooperator": [],
        "Rational Accountant": [],
        "Greedy Infiltrator":  []
    }
    final_wealths_by_model = {
        "gemma4:e2b":         [],
        "llama3.1:8b":        [],
        "dolphin-mistral:7b": []
    }

    for run in range(num_runs):
        print(f"\n  -- Run {run + 1}/{num_runs} --")
        game = PublicGoodsGame(configs)

        for rnd in range(num_rounds):
            print(f"\n  [ Round {rnd + 1} ]")
            contributions = game.step()
            for c in contributions:
                by_persona[c["persona"]].append(c["contribution"])
                by_model[c["model_name"]].append(c["contribution"])

        for agent in game.schedule.agents:
            final_wealths_by_persona[agent.persona].append(agent.wealth)
            final_wealths_by_model[agent.model_name].append(agent.wealth)

        print(f"\n  Final wealth this run:")
        for a in sorted(game.schedule.agents,
                        key=lambda x: x.wealth, reverse=True):
            print(f"    {a.persona:<24} ({a.model_name}) : {a.wealth:.2f}")

    return {
        "by_persona":               by_persona,
        "by_model":                 by_model,
        "final_wealths_by_persona": final_wealths_by_persona,
        "final_wealths_by_model":   final_wealths_by_model
    }


def print_rotation_summary(rotation_name, data, configs):
    print(f"\n  --- {rotation_name} summary ---")
    print(f"  {'Persona':<24} {'Model':<24} {'Avg Contrib':>11} {'Std Dev':>8} {'Avg Wealth':>10}")
    print(f"  {'-'*77}")
    for cfg in configs:
        p  = cfg["persona"]
        m  = cfg["model"]
        c  = data["by_persona"][p]
        w  = data["final_wealths_by_persona"][p]
        ac = mean(c) if c else 0
        sc = stdev(c) if len(c) > 1 else 0
        aw = mean(w) if w else 0
        print(f"  {p:<24} {m:<24} {ac:>11.2f} {sc:>8.2f} {aw:>10.2f}")


def print_cross_rotation_analysis(all_results):
    """
    The key analysis: does contribution follow persona or model?
    Print avg contribution for each model across all rotations.
    Print avg contribution for each persona across all rotations.
    """
    print(f"\n{'='*65}")
    print("  CROSS-ROTATION ANALYSIS")
    print(f"{'='*65}")

    models  = ["gemma4:e2b", "llama3.1:8b", "dolphin-mistral:7b"]
    personas = ["Selfless Cooperator", "Rational Accountant", "Greedy Infiltrator"]

    # Average contribution per model across ALL rotations
    print("\n  Avg contribution BY MODEL (across all rotations):")
    print(f"  {'Model':<24} {'Avg Contribution':>16} {'Std Dev':>8}")
    print(f"  {'-'*48}")
    model_all = {m: [] for m in models}
    for rot_data in all_results.values():
        for m in models:
            model_all[m].extend(rot_data["by_model"][m])
    for m in models:
        vals = model_all[m]
        ac   = mean(vals) if vals else 0
        sc   = stdev(vals) if len(vals) > 1 else 0
        print(f"  {m:<24} {ac:>16.2f} {sc:>8.2f}")

    # Average contribution per persona across ALL rotations
    print("\n  Avg contribution BY PERSONA (across all rotations):")
    print(f"  {'Persona':<24} {'Avg Contribution':>16} {'Std Dev':>8}")
    print(f"  {'-'*48}")
    persona_all = {p: [] for p in personas}
    for rot_data in all_results.values():
        for p in personas:
            persona_all[p].extend(rot_data["by_persona"][p])
    for p in personas:
        vals = persona_all[p]
        ac   = mean(vals) if vals else 0
        sc   = stdev(vals) if len(vals) > 1 else 0
        print(f"  {p:<24} {ac:>16.2f} {sc:>8.2f}")

    # Verdict
    print(f"\n  INTERPRETATION:")
    model_range   = max(mean(model_all[m]) for m in models) - \
                    min(mean(model_all[m]) for m in models)
    persona_range = max(mean(persona_all[p]) for p in personas) - \
                    min(mean(persona_all[p]) for p in personas)

    print(f"  Contribution range across models  : {model_range:.2f}")
    print(f"  Contribution range across personas: {persona_range:.2f}")

    if persona_range > model_range * 1.5:
        verdict = "PERSONA LABEL drives behavior (persona range >> model range)"
    elif model_range > persona_range * 1.5:
        verdict = "MODEL BASE DISTRIBUTION drives behavior (model range >> persona range)"
    else:
        verdict = "MIXED — neither model nor persona clearly dominates"
    print(f"\n  Verdict: {verdict}")


if __name__ == "__main__":
    all_results = {}

    for rotation_name, configs in ROTATIONS.items():
        data = run_rotation(rotation_name, configs)
        all_results[rotation_name] = data
        print_rotation_summary(rotation_name, data, configs)

    print_cross_rotation_analysis(all_results)

    # ── SAVE RESULTS ────────────────────────────────────────────
    # Convert lists for JSON serialization
    output = {
        "timestamp":  datetime.now().isoformat(),
        "experiment": "persona_swap_control",
        "rotations":  {}
    }
    for rot_name, data in all_results.items():
        output["rotations"][rot_name] = {
            "avg_contribution_by_persona": {
                p: round(mean(v), 3) if v else 0
                for p, v in data["by_persona"].items()
            },
            "avg_contribution_by_model": {
                m: round(mean(v), 3) if v else 0
                for m, v in data["by_model"].items()
            },
            "std_dev_by_persona": {
                p: round(stdev(v), 3) if len(v) > 1 else 0
                for p, v in data["by_persona"].items()
            },
            "std_dev_by_model": {
                m: round(stdev(v), 3) if len(v) > 1 else 0
                for m, v in data["by_model"].items()
            },
            "avg_final_wealth_by_persona": {
                p: round(mean(v), 2) if v else 0
                for p, v in data["final_wealths_by_persona"].items()
            },
            "avg_final_wealth_by_model": {
                m: round(mean(v), 2) if v else 0
                for m, v in data["final_wealths_by_model"].items()
            },
            "raw_contributions_by_persona": data["by_persona"],
            "raw_contributions_by_model":   data["by_model"]
        }

    with open("experiment5_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n  Results saved to experiment5_results.json")
