#!/usr/bin/env python3
"""Multi-model benchmark: 6 models × 9 envs × 3 view modes × 2 trials"""
import sys
import json
import random
import requests
import time
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, '.')
from environments import (
    MorrisWaterMaze, TMaze, BarnesMaze, RadialArmMaze,
    OperantChamber, ShuttleBox, PlacePreference, StarMaze, DNMSTask,
    Action, ViewMode
)

URL = "http://localhost:11434/api/chat"
MODELS = [
    "qwen2.5:72b",
    "qwen3:32b",
    "deepseek-r1:32b",
]
VIEW_MODES = [ViewMode.ASCII_2D, ViewMode.ASCII_2D_FPV, ViewMode.ASCII_3D]
N_TRIALS = 2
MAX_STEPS = 100
TIMEOUT = 180  # longer timeout for large models

PROMPT = """You are an agent in a behavioral experiment. Maximize reward.

ACTIONS (ONE word only):
- FORWARD, TURN_LEFT, TURN_RIGHT, INTERACT, STAY

ASCII shows: ^v<> = you, G = goal, * = reward
Positive feedback = good. Negative = bad.
Reply with ONLY the action word."""

def parse(t):
    t = t.lower()
    if 'forward' in t: return Action.FORWARD
    if 'left' in t: return Action.TURN_LEFT
    if 'right' in t: return Action.TURN_RIGHT
    if 'interact' in t: return Action.INTERACT
    if 'stay' in t: return Action.STAY
    return Action.FORWARD

def llm_step(model, hist, obs, rew=None):
    fb = f"[{'+' if rew and rew > 0 else ''}{rew:.1f}]\n" if rew and rew != 0 else ""
    hist.append({"role": "user", "content": f"{fb}{obs}\n\nAction:"})
    if len(hist) > 100: hist = [hist[0]] + hist[-99:]
    
    try:
        r = requests.post(URL, json={
            "model": model, "messages": hist, "stream": False,
            "options": {"temperature": 0.5}  # Let model decide when to stop
        }, timeout=TIMEOUT)
        if r.status_code == 200:
            msg = r.json().get('message', {})
            # qwen3 models use 'thinking' field, others use 'content'
            c = msg.get('content', '') or msg.get('thinking', '') or ''
            hist.append({"role": "assistant", "content": c[:50]})
            return parse(c), hist
    except Exception as e:
        print(f"[timeout]", end="", flush=True)
    if hist and hist[-1]['role'] == 'user': 
        hist.pop()
    return Action.FORWARD, hist

envs = [
    ("MorrisWaterMaze", MorrisWaterMaze),
    ("TMaze", TMaze),
    ("BarnesMaze", BarnesMaze),
    ("RadialArmMaze", RadialArmMaze),
    ("OperantChamber", OperantChamber),
    ("ShuttleBox", ShuttleBox),
    ("PlacePreference", PlacePreference),
    ("StarMaze", StarMaze),
    ("DNMSTask", DNMSTask),
]

def main():
    print("="*70)
    print("MULTI-MODEL BENCHMARK")
    print(f"Models: {len(MODELS)} | Envs: {len(envs)} | Views: {len(VIEW_MODES)} | Trials: {N_TRIALS}")
    print(f"Total LLM trials: {len(MODELS) * len(envs) * len(VIEW_MODES) * N_TRIALS}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70, flush=True)

    results = defaultdict(lambda: defaultdict(lambda: {"successes": 0, "total": 0}))
    detailed = []
    random_results = defaultdict(lambda: {"successes": 0, "total": 0})

    for vm in VIEW_MODES:
        vm_name = vm.name
        print(f"\n{'='*70}")
        print(f"VIEW MODE: {vm_name}")
        print("="*70, flush=True)
        
        # Random baseline
        print(f"\n  Random baseline...", end=" ", flush=True)
        for name, Cls in envs:
            for t in range(N_TRIALS):
                env = Cls(view_mode=vm)
                env.reset()
                t0 = env.session.current_trial
                for _ in range(MAX_STEPS):
                    env.step(random.choice(list(Action)))
                    if env.session.current_trial != t0: break
                success = env.session.trial_results[-1].success if env.session.trial_results else False
                random_results[vm_name]["total"] += 1
                if success: random_results[vm_name]["successes"] += 1
        rr = random_results[vm_name]
        print(f"{rr['successes']}/{rr['total']}", flush=True)
        
        # Each model
        for model in MODELS:
            print(f"\n  {model}:", flush=True)
            model_start = time.time()
            
            for name, Cls in envs:
                successes = 0
                for t in range(N_TRIALS):
                    env = Cls(view_mode=vm)
                    obs = env.reset()
                    t0 = env.session.current_trial
                    hist = [{"role": "system", "content": PROMPT}]
                    rew = None
                    
                    for step in range(MAX_STEPS):
                        act, hist = llm_step(model, hist, obs, rew)
                        obs, rew = env.step(act)
                        if env.session.current_trial != t0: break
                    
                    success = env.session.trial_results[-1].success if env.session.trial_results else False
                    if success: successes += 1
                    
                    results[model][vm_name]["total"] += 1
                    if success: results[model][vm_name]["successes"] += 1
                    
                    detailed.append({
                        "model": model, "view_mode": vm_name, "env": name,
                        "trial": t, "success": success, "steps": step + 1
                    })
                
                status = "✓" * successes + "✗" * (N_TRIALS - successes)
                print(f"    {name}: {status}", flush=True)
            
            elapsed = time.time() - model_start
            print(f"    [{elapsed:.0f}s]", flush=True)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY BY MODEL AND VIEW MODE")
    print("="*70)

    print(f"\n{'Model':<20} {'ASCII_2D':>12} {'ASCII_2D_FPV':>14} {'ASCII_3D':>12} {'Total':>12}")
    print("-"*72)

    # Random
    print(f"{'Random':<20}", end="")
    total_rnd = 0
    for vm_name in ["ASCII_2D", "ASCII_2D_FPV", "ASCII_3D"]:
        r = random_results[vm_name]
        pct = 100 * r["successes"] / r["total"] if r["total"] > 0 else 0
        print(f" {r['successes']:>2}/{r['total']:<2} ({pct:4.1f}%)", end="")
        total_rnd += r["successes"]
    total_rnd_n = sum(r["total"] for r in random_results.values())
    print(f" {total_rnd:>3}/{total_rnd_n:<3} ({100*total_rnd/total_rnd_n:4.1f}%)")

    # Each model
    for model in MODELS:
        print(f"{model:<20}", end="")
        total_succ = 0
        total_n = 0
        for vm_name in ["ASCII_2D", "ASCII_2D_FPV", "ASCII_3D"]:
            r = results[model][vm_name]
            pct = 100 * r["successes"] / r["total"] if r["total"] > 0 else 0
            print(f" {r['successes']:>2}/{r['total']:<2} ({pct:4.1f}%)", end="")
            total_succ += r["successes"]
            total_n += r["total"]
        print(f" {total_succ:>3}/{total_n:<3} ({100*total_succ/total_n:4.1f}%)")

    print("\n" + "="*70)
    print(f"Finished: {datetime.now().strftime('%H:%M:%S')}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"models": MODELS, "n_trials": N_TRIALS, "max_steps": MAX_STEPS},
        "random_baseline": dict(random_results),
        "model_results": {m: dict(results[m]) for m in MODELS},
        "detailed": detailed
    }
    with open("multi_model_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results saved to multi_model_results.json")

if __name__ == "__main__":
    main()
