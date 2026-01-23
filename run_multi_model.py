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
    "gemma3:27b",  # fast non-reasoning model
]
VIEW_MODES = [ViewMode.ASCII_2D, ViewMode.ASCII_2D_FPV, ViewMode.ASCII_3D]
N_TRIALS = 2
MAX_STEPS = 100
TIMEOUT = 30  # shorter timeout

# Log file path for continuous logging
LLM_LOG_FILE = "llm_interaction_logs.txt"
_log_count = 0

def append_log(log_entry):
    """Append a log entry to the text file immediately with clear formatting."""
    global _log_count
    with open(LLM_LOG_FILE, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"ENTRY #{_log_count + 1}  |  {log_entry['timestamp']}\n")
        f.write(f"{'='*80}\n")
        ctx = log_entry.get('context', {})
        f.write(f"Model: {log_entry['model']}\n")
        f.write(f"Env: {ctx.get('env', '?')}  |  View: {ctx.get('view_mode', '?')}  |  Trial: {ctx.get('trial', '?')}  |  Step: {ctx.get('step', '?')}\n")
        f.write(f"-"*80 + "\n")
        if log_entry.get('reward_feedback'):
            f.write(f"REWARD: {log_entry['reward_feedback'].strip()}\n")
        f.write(f"OBSERVATION:\n{log_entry['observation']}\n")
        f.write(f"-"*80 + "\n")
        if log_entry.get('thinking'):
            f.write(f"THINKING:\n{log_entry['thinking']}\n")
            f.write(f"-"*80 + "\n")
        f.write(f"RESPONSE: {log_entry.get('response', '')}\n")
        f.write(f"PARSED ACTION: {log_entry.get('parsed_action', 'N/A')}\n")
        if log_entry.get('error'):
            f.write(f"ERROR: {log_entry['error']}\n")
    _log_count += 1

def finalize_log():
    """Add final summary to log file."""
    with open(LLM_LOG_FILE, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"END OF LOG - {_log_count} entries\n")
        f.write(f"{'='*80}\n")

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

ACTION_REMINDER = """Actions: FORWARD, TURN_LEFT, TURN_RIGHT, INTERACT, STAY
Reply with ONLY the action word."""

def llm_step(model, hist, obs, rew=None, log_context=None):
    # Show reward with enough precision (2 decimal places)
    fb = f"[{'+' if rew and rew > 0 else ''}{rew:.2f}]\n" if rew and rew != 0 else ""
    user_content = f"{fb}{obs}\n\nAction:"
    hist.append({"role": "user", "content": user_content})
    if len(hist) > 200: hist = [hist[0]] + hist[-199:]  # Keep last 100 actions (200 messages = 100 turns)
    
    # Build messages with action reminder at the very end
    messages_to_send = hist + [{"role": "user", "content": ACTION_REMINDER}]
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "context": log_context or {},
        "input": user_content,
        "reward_feedback": fb if fb else None,
        "observation": obs,
        "response": None,
        "thinking": None,
        "parsed_action": None,
        "error": None,
    }
    
    try:
        r = requests.post(URL, json={
            "model": model, "messages": messages_to_send, "stream": False,
            "options": {"temperature": 0.5}  # Let model decide when to stop
        }, timeout=TIMEOUT)
        if r.status_code == 200:
            msg = r.json().get('message', {})
            # qwen3 models use 'thinking' field, others use 'content'
            content = msg.get('content', '') or ''
            thinking = msg.get('thinking', '') or ''
            c = content or thinking
            
            log_entry["response"] = content
            log_entry["thinking"] = thinking if thinking else None
            parsed = parse(c)
            log_entry["parsed_action"] = parsed.name
            
            hist.append({"role": "assistant", "content": c[:50]})
            append_log(log_entry)
            return parsed, hist
    except Exception as e:
        log_entry["error"] = str(e)
        print(f"[timeout]", end="", flush=True)
    
    append_log(log_entry)
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
    global _log_count
    # Initialize log file (clear previous content)
    _log_count = 0
    with open(LLM_LOG_FILE, "w") as f:
        f.write("")  # Start fresh
    print(f"LLM logs will be written continuously to {LLM_LOG_FILE}")
    
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
                        log_ctx = {"env": name, "view_mode": vm_name, "trial": t, "step": step}
                        act, hist = llm_step(model, hist, obs, rew, log_context=log_ctx)
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
    
    # Finalize the JSON log file
    finalize_log()
    print(f"LLM interaction logs saved to {LLM_LOG_FILE} ({_log_count} entries)")

if __name__ == "__main__":
    main()
