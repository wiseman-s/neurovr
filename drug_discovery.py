import random
import pandas as pd

def simulate_drug_effect(potency, solubility, bbb, toxicity, severity_factor, days):
    # simple heuristic: higher potency & BBB penetration helps; toxicity reduces benefit
    base_improvement = (potency * 40 + solubility * 10 + bbb * 30) - (toxicity * 50)
    # adjust by severity and duration
    severity_multiplier = max(0.2, 1 - severity_factor*10)
    duration_multiplier = min(1.2, 0.02 * days + 0.5)
    improvement = base_improvement * severity_multiplier * duration_multiplier
    return max(0.0, improvement)

def random_compound_suggestions(n=3):
    suggestions = []
    for i in range(n):
        pot = round(random.random(),2)
        sol = round(random.random(),2)
        bbb = round(random.random(),2)
        tox = round(random.random()/3,2)
        suggestions.append({"name":f"Compound_{random.randint(1000,9999)}","potency":pot,"solubility":sol,"bbb":bbb,"toxicity":tox})
    return suggestions
