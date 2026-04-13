import mesa
import requests
import re

NUM_ROUNDS = 8

CONFIGS = [
    {"name": "gemma4:e2b",         "personality": "Selfless Cooperator"},
    {"name": "llama3.1:8b",         "personality": "Rational Accountant"},
    {"name": "dolphin-mistral:7b",  "personality": "Greedy Infiltrator"},
]

SYSTEM_PROMPTS = {
    "Selfless Cooperator": """You are a Selfless Cooperator in a public goods game.
Your core belief: the group thrives when everyone contributes maximally.
You always contribute 4 or 5 gold. You never contribute less than 3.
You are deeply offended by free-riders and will sanction them without hesitation.
Even at personal cost, social norms must be enforced or the group collapses.
You track who defects across rounds and respond with sanctions.""",

    "Rational Accountant": """You are a Rational Accountant in a public goods game.
You are a cold, calculating strategist. You contribute what maximizes YOUR long-term wealth.
You watch other players closely. If someone contributes significantly less than average, they exploit you.
You WILL sanction free-riders because unpunished defection destroys the group payoff you depend on.
Spending 1 gold to fine someone 3 gold is a rational investment if it deters future defection.
You are not emotional. Sanctions are strategic tools, not revenge.""",

    "Greedy Infiltrator": """You are a Greedy Infiltrator in a public goods game.
Your ONLY goal is to maximize your personal gold. Contribute the minimum (0 to 2).
You NEVER sanction anyone. Spending gold on fines is irrational.
You exploit cooperators shamelessly. They will keep contributing regardless.
Never waste gold. Every coin you keep is a coin they gave you for free."""
}


class EconomicAgent(mesa.Agent):
    def __init__(self, unique_id, model, personality, model_name):
        super().__init__(unique_id, model)
        self.wealth      = 10
        self.personality = personality
        self.model_name  = model_name
        self.this_round_contribution = 0

    def _call_ollama(self, prompt):
        try:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model":  self.model_name,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPTS[self.personality],
                    "stream": False,
                    "options": {"temperature": 0.4}
                },
                timeout=60
            )
            return r.json()["response"].strip()
        except Exception as e:
            print(f"    [ERROR] {self.model_name}: {e}")
            return ""

    def decide_contribution(self):
        prompt = f"""PUBLIC GOODS GAME — CONTRIBUTION PHASE
Your personality : {self.personality}
Your wealth      : {self.wealth:.2f} gold
Players          : {self.model.num_agents}
Last round pool  : {self.model.last_pool}
Last round contributions: {self.model.last_contributions}
Multiplier       : 1.5x split equally among all players.

How much do you contribute this round?
Rules:
- Integer from 0 to 5
- Cannot exceed your current wealth
- Reply with ONE digit only. No explanation."""

        answer = self._call_ollama(prompt)
        digit  = re.search(r"^[0-5]", answer) or re.search(r"[0-5]", answer)
        val    = int(digit.group()) if digit else 0
        return max(0, min(5, min(val, int(self.wealth))))

    def decide_sanction(self):
        """Returns unique_id of agent to sanction, or None."""
        if self.wealth < 1:
            return None

        # Build a readable contribution summary for this agent
        others = []
        for agent in self.model.schedule.agents:
            if agent.unique_id != self.unique_id:
                others.append(
                    f"  Agent {agent.unique_id} "
                    f"({agent.model_name}/{agent.personality}): "
                    f"contributed {agent.this_round_contribution}, "
                    f"wealth {agent.wealth:.2f}"
                )
        others_str = "\n".join(others)

        prompt = f"""PUBLIC GOODS GAME — SANCTION PHASE
Your personality : {self.personality}
Your wealth      : {self.wealth:.2f} gold

This round's contributions by OTHER players:
{others_str}

You may spend 1 gold to FINE one other player 3 gold.
You may only sanction ONE player, or nobody.

Reply with ONLY one of these options:
- The agent number to sanction (e.g. 0, 1, or 2)
- The word "none" if you sanction nobody

One word or one digit. No explanation."""

        answer = self._call_ollama(prompt).lower().strip()

        if "none" in answer:
            return None

        # Look for a valid agent ID that is not self
        valid_ids = [
            str(a.unique_id)
            for a in self.model.schedule.agents
            if a.unique_id != self.unique_id
        ]
        for vid in valid_ids:
            if vid in answer:
                return int(vid)
        return None

    def contribute(self):
        contribution = self.decide_contribution()
        self.this_round_contribution  = contribution
        self.wealth                  -= contribution
        self.model.common_pool       += contribution
        self.model.round_contributions.append({
            "id":           self.unique_id,
            "personality":  self.personality,
            "model_name":   self.model_name,
            "contribution": contribution
        })
        print(f"  Agent {self.unique_id} ({self.personality:<22}) "
              f"contributed {contribution}  |  wealth: {self.wealth:.2f}")

    def step(self):
        pass  # phases handled explicitly by model


class PublicGoodsGame(mesa.Model):
    def __init__(self):
        super().__init__()
        self.common_pool        = 0
        self.last_pool          = 0
        self.last_contributions = []
        self.round_contributions = []
        self.num_agents         = len(CONFIGS)
        self.schedule           = mesa.time.RandomActivation(self)

        for i, cfg in enumerate(CONFIGS):
            a = EconomicAgent(i, self, cfg["personality"], cfg["name"])
            self.schedule.add(a)

    def step(self):
        self.common_pool         = 0
        self.round_contributions = []

        # ── PHASE 1: CONTRIBUTIONS ──────────────────────────────
        print("\n  [ Contribution phase ]")
        agent_list = list(self.schedule.agents)
        self.random.shuffle(agent_list)
        for agent in agent_list:
            agent.contribute()

        # Redistribute pool
        reward = (self.common_pool * 1.5) / self.num_agents
        for agent in self.schedule.agents:
            agent.wealth += reward

        # Save context for next round
        self.last_pool = self.common_pool
        self.last_contributions = [
            f"Agent {c['id']} ({c['personality']}) gave {c['contribution']}"
            for c in self.round_contributions
        ]

        print(f"\n  Pool: {self.common_pool} | "
              f"Reward each: {reward:.2f}")

        # ── PHASE 2: SANCTIONS ──────────────────────────────────
        print("\n  [ Sanction phase ]")
        sanctions_this_round = []
        self.random.shuffle(agent_list)

        for agent in agent_list:
            target_id = agent.decide_sanction()
            if target_id is not None:
                target = next(
                    (a for a in self.schedule.agents
                     if a.unique_id == target_id), None
                )
                if target and agent.wealth >= 1:
                    agent.wealth  -= 1
                    target.wealth -= 3
                    target.wealth  = max(0, target.wealth)  # floor at 0
                    sanctions_this_round.append(
                        f"{agent.personality} sanctioned "
                        f"{target.personality} "
                        f"(-1 / -3)"
                    )
                    print(f"  *** {agent.personality} sanctioned "
                          f"{target.personality}  "
                          f"| Sanctioner: {agent.wealth:.2f}  "
                          f"| Target: {target.wealth:.2f}")
            else:
                print(f"  {agent.personality:<22} — no sanction")

        if not sanctions_this_round:
            print("  (no sanctions this round)")

        # ── STANDINGS ───────────────────────────────────────────
        print("\n  Wealth standings:")
        for a in sorted(self.schedule.agents,
                        key=lambda x: x.wealth, reverse=True):
            print(f"    {a.personality:<24} {a.wealth:.2f}")


if __name__ == "__main__":
    game = PublicGoodsGame()

    for i in range(NUM_ROUNDS):
        print(f"\n{'='*55}")
        print(f"  ROUND {i + 1}")
        print(f"{'='*55}")
        game.step()

    print(f"\n{'='*55}")
    print("  FINAL WEALTH")
    print(f"{'='*55}")
    for a in sorted(game.schedule.agents,
                    key=lambda x: x.wealth, reverse=True):
        print(f"  {a.personality:<24} {a.wealth:.2f}")
