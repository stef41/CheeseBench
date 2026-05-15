"""Generate publication-quality figures for the ALIFE paper."""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- Data ---
# Multi-model results (ASCII_2D view mode)
ENVS = ['BarnesMaze', 'DNMSTask', 'MorrisWaterMaze', 'OperantChamber',
        'PlacePreference', 'RadialArmMaze', 'ShuttleBox', 'StarMaze', 'TMaze']

ENV_SHORT = ['Barnes', 'DNMS', 'MWM', 'Operant', 'CPP', 'RAM', 'Shuttle', 'Star', 'T-Maze']

MODELS = {
    # Per-env best across the three ASCII view modes, from leaderboard.json (schema v3).
    # Order: Barnes, DNMS, MWM, Operant, CPP, RAM, Shuttle, Star, T-Maze
    'Qwen2.5-VL-3B':  [37.5, 38.0,  95.0,  54.0, 66.7, 0.0,  67.5,  5.0, 100.0],
    'Qwen2.5-VL-7B':  [75.0, 54.0,  95.0,  84.0, 91.7, 10.0, 80.0, 35.0,  87.5],
    'Qwen2.5-VL-32B': [68.8, 52.0,  95.0, 100.0, 66.7, 0.0,  95.0, 22.5,  82.5],
    'Qwen2.5-VL-72B': [43.8, 48.0,  85.0,  98.0, 66.7, 0.0,  70.0, 42.5,  72.5],
    'InternVL2.5-8B': [56.2, 48.0,  60.0,  22.0, 83.3, 5.0,  22.5, 10.0,  40.0],
    'Phi-4-MM-14B':   [25.0, 54.0,  85.0,  60.0, 83.3, 0.0,  50.0, 12.5,  57.5],
}

# Random baseline averaged across all 6 model runs (best of 3 ASCII view modes per env)
RANDOM = [43.8, 53.2, 46.7, 38.3, 76.4, 0.0, 32.9, 7.1, 43.3]

# Tabular-QL: simple RL agent (Q-learning + state hashing)
CSCG = [0.0, 58.0, 0.0, 98.0, 25.0, 0.0, 100.0, 0.0, 0.0]

# BFS oracle: parses ASCII grid, BFS to visible goals
BFS = [0.0, 54.0, 100.0, 42.0, 0.0, 0.0, 0.0, 45.0, 0.0]

ANIMAL = [80, 80, 85, 90, 75, 70, 70, 80, 80]

# Cognitive dimensions for each env
COG_DIM = ['Spatial', 'WM', 'Spatial', 'Instr.Cond.', 'Assoc.', 'WM', 'Avoidance', 'Spatial', 'Ego.Nav.']

# Ablation data
HISTORY = {'1': 57.0, '3': 46.7, '5': 31.1, '10': 26.7}
PROMPT = {'Default': 46.7, 'Minimal': 35.6, 'CoT': 31.9, 'Few-shot': 29.6}
ACTIONS = {'1': 28.9, '4': 51.9, '8': 45.9, '16': 20.0}
VISION = {'LLM-7B': 41.5, 'Text-7B': 21.5, 'LLM-32B': 32.6, 'Text-32B': 43.0}

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLORS = {
    'Qwen2.5-VL-3B':  '#1f77b4',
    'Qwen2.5-VL-7B':  '#2ca02c',
    'Qwen2.5-VL-32B': '#ff7f0e',
    'Qwen2.5-VL-72B': '#d62728',
    'InternVL2.5-8B':  '#9467bd',
    'Phi-4-MM-14B':    '#8c564b',
}


def fig1_main_results():
    """Grouped bar chart: per-env success rates for all models + baselines + animal."""
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    
    # Trial counts per env (same order as ENVS)
    TRIAL_N = [16, 100, 20, 50, 12, 20, 40, 40, 40]
    
    def binom_se(pcts, ns):
        """Binomial standard error for percentage values."""
        return [np.sqrt((p/100)*(1-p/100)/n)*100 for p, n in zip(pcts, ns)]
    
    n_envs = len(ENVS)
    n_models = len(MODELS)
    # Total bars: random + CSCG + n_models = n_models + 2
    n_bars = n_models + 2
    bar_w = 0.08
    x = np.arange(n_envs)
    
    # Random baseline
    ax.bar(x - (n_bars/2 - 0.5)*bar_w, RANDOM, bar_w, color='#cccccc',
           edgecolor='#999999', linewidth=0.5, label='Random', zorder=2,
           yerr=binom_se(RANDOM, TRIAL_N), error_kw=dict(lw=0.5, capsize=1.5))
    
    # Tabular-QL baseline
    ax.bar(x - (n_bars/2 - 1.5)*bar_w, CSCG, bar_w, color='#17becf',
           edgecolor='white', linewidth=0.3, label='Tabular-QL', zorder=2,
           yerr=binom_se(CSCG, TRIAL_N), error_kw=dict(lw=0.5, capsize=1.5))
    
    # Models
    for i, (name, vals) in enumerate(MODELS.items()):
        offset = x - (n_bars/2 - 2.5 - i)*bar_w
        ax.bar(offset, vals, bar_w, color=COLORS[name],
               edgecolor='white', linewidth=0.3, label=name, zorder=2,
               yerr=binom_se(vals, TRIAL_N), error_kw=dict(lw=0.5, capsize=1.5))
    
    # Animal baseline as markers
    ax.scatter(x, ANIMAL, marker='D', s=25, color='black', zorder=3, label='Rodent baseline')
    
    # Annotate environments where all bars are near-zero so they don't look like missing data
    ram_idx = ENVS.index('RadialArmMaze')
    ax.annotate('all ≤5%', xy=(x[ram_idx], 5), xytext=(x[ram_idx], 12),
                ha='center', fontsize=5.5, color='#666666',
                arrowprops=dict(arrowstyle='->', color='#999999', lw=0.5))
    
    ax.set_xticks(x)
    ax.set_xticklabels(ENV_SHORT, rotation=30, ha='right')
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(0, 105)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.28), frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=50, color='#dddddd', linestyle='--', linewidth=0.5, zorder=1)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    
    fig.savefig('fig_main_results.pdf')
    fig.savefig('fig_main_results.png')
    plt.close(fig)
    print('Created fig_main_results.pdf')


def fig2_radar():
    """Cognitive profile radar: best VLM vs rodent baseline."""
    # Aggregate by cognitive dimension
    cog_dims = ['Spatial Learning', 'Working Memory', 'Ego. Navigation',
                'Instr. Cond.', 'Avoidance Learning', 'Assoc. Learning']
    
    # Map envs to cog dims
    env_to_cog = {
        'BarnesMaze': 'Spatial Learning', 'MorrisWaterMaze': 'Spatial Learning',
        'StarMaze': 'Spatial Learning', 'DNMSTask': 'Working Memory',
        'RadialArmMaze': 'Working Memory', 'TMaze': 'Ego. Navigation',
        'OperantChamber': 'Instr. Cond.', 'ShuttleBox': 'Avoidance Learning',
        'PlacePreference': 'Assoc. Learning',
    }
    
    # Best VLM per env (Qwen2.5-VL-72B as representative for cognitive profile)
    best_vlm = MODELS['Qwen2.5-VL-72B']
    cscg_vals_raw = CSCG
    
    # Average by cog dim
    vlm_by_cog = {d: [] for d in cog_dims}
    animal_by_cog = {d: [] for d in cog_dims}
    random_by_cog = {d: [] for d in cog_dims}
    cscg_by_cog = {d: [] for d in cog_dims}
    
    for i, env in enumerate(ENVS):
        cog = env_to_cog[env]
        vlm_by_cog[cog].append(best_vlm[i])
        animal_by_cog[cog].append(ANIMAL[i])
        random_by_cog[cog].append(RANDOM[i])
        cscg_by_cog[cog].append(cscg_vals_raw[i])
    
    vlm_vals = [np.mean(vlm_by_cog[d]) for d in cog_dims]
    animal_vals = [np.mean(animal_by_cog[d]) for d in cog_dims]
    random_vals = [np.mean(random_by_cog[d]) for d in cog_dims]
    cscg_vals = [np.mean(cscg_by_cog[d]) for d in cog_dims]
    
    # Radar - rotate start so labels avoid data-point overlap at top
    angles = np.linspace(0, 2*np.pi, len(cog_dims), endpoint=False)
    # Rotate 30° so no label sits exactly at top (where high values crowd)
    angles = (angles + np.pi/6).tolist()
    vlm_vals_c = vlm_vals + [vlm_vals[0]]
    animal_vals_c = animal_vals + [animal_vals[0]]
    random_vals_c = random_vals + [random_vals[0]]
    cscg_vals_c = cscg_vals + [cscg_vals[0]]
    angles_c = angles + [angles[0]]
    
    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
    
    ax.plot(angles_c, animal_vals_c, 'o-', color='#2ca02c', linewidth=1.5,
            markersize=4, label='Rodent', zorder=3)
    ax.fill(angles_c, animal_vals_c, alpha=0.1, color='#2ca02c')
    
    ax.plot(angles_c, vlm_vals_c, 's-', color='#d62728', linewidth=1.5,
            markersize=4, label='Qwen2.5-VL-72B', zorder=3)
    ax.fill(angles_c, vlm_vals_c, alpha=0.1, color='#d62728')
    
    ax.plot(angles_c, cscg_vals_c, 'D--', color='#17becf', linewidth=1.0,
            markersize=3, label='Tabular-QL', zorder=2)
    
    ax.plot(angles_c, random_vals_c, '^--', color='#999999', linewidth=1.0,
            markersize=3, label='Random', zorder=2)
    
    ax.set_thetagrids(np.degrees(angles), cog_dims)
    ax.set_ylim(0, 110)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], fontsize=6)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.2), frameon=False)
    # Pad tick labels away from the outer ring
    ax.tick_params(pad=15)
    
    fig.savefig('fig_cognitive_radar.pdf', bbox_inches='tight', pad_inches=0.15)
    fig.savefig('fig_cognitive_radar.png', bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print('Created fig_cognitive_radar.pdf')


def fig3_ablations():
    """2x2 ablation panel: history, prompt, actions, vision."""
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.0))
    
    # History
    ax = axes[0, 0]
    ks = list(HISTORY.keys())
    vs = list(HISTORY.values())
    bars = ax.bar(ks, vs, color='#1f77b4', edgecolor='white', width=0.5)
    ax.set_xlabel('History Length (observation-action pairs)')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('(a) History Length')
    ax.set_ylim(0, 70)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, v in zip(bars, vs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5, f'{v:.0f}%',
                ha='center', va='bottom', fontsize=7)
    
    # Prompt
    ax = axes[0, 1]
    ks = list(PROMPT.keys())
    vs = list(PROMPT.values())
    bars = ax.bar(ks, vs, color='#2ca02c', edgecolor='white', width=0.5)
    ax.set_xlabel('Prompt Variant')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('(b) Prompt Strategy')
    ax.set_ylim(0, 60)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, v in zip(bars, vs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.0, f'{v:.0f}%',
                ha='center', va='bottom', fontsize=7)
    
    # Actions
    ax = axes[1, 0]
    ks = list(ACTIONS.keys())
    vs = list(ACTIONS.values())
    bars = ax.bar(ks, vs, color='#ff7f0e', edgecolor='white', width=0.5)
    ax.set_xlabel('Actions per Call (k)')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('(c) Action Batch Size')
    ax.set_ylim(0, 65)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, v in zip(bars, vs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.0, f'{v:.0f}%',
                ha='center', va='bottom', fontsize=7)
    
    # Vision
    ax = axes[1, 1]
    labels = ['LLM\n7B', 'Text\n7B', 'LLM\n32B', 'Text\n32B']
    vs = list(VISION.values())
    colors = ['#d62728', '#999999', '#d62728', '#999999']
    bars = ax.bar(labels, vs, color=colors, edgecolor='white', width=0.5)
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('(d) Vision vs. Text-Only')
    ax.set_ylim(0, 60)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, v in zip(bars, vs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.0, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    fig.savefig('fig_ablations.pdf')
    fig.savefig('fig_ablations.png')
    plt.close(fig)
    print('Created fig_ablations.pdf')


def fig4_scaling():
    """Scaling curve: model size vs overall success rate."""
    sizes = [3, 7, 32, 72]
    qwen_scores = [
        np.mean(MODELS['Qwen2.5-VL-3B']),
        np.mean(MODELS['Qwen2.5-VL-7B']),
        np.mean(MODELS['Qwen2.5-VL-32B']),
        np.mean(MODELS['Qwen2.5-VL-72B']),
    ]
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(sizes, qwen_scores, 'o-', color='#d62728', linewidth=1.5, markersize=6, label='Qwen2.5-VL')
    
    # Add other models as points
    other = {
        'InternVL2.5-8B': (8, np.mean(MODELS['InternVL2.5-8B'])),
        'Phi-4-MM-14B': (14, np.mean(MODELS['Phi-4-MM-14B'])),
    }
    for name, (sz, sc) in other.items():
        ax.scatter(sz, sc, s=40, color=COLORS[name], zorder=3, label=name, marker='s')
    
    # Random baseline
    rmean = np.mean(RANDOM)
    ax.axhline(y=rmean, color='#999999', linestyle='--', linewidth=1, label=f'Random ({rmean:.1f}%)')
    
    ax.set_xlabel('Model Size (B parameters)')
    ax.set_ylabel('Overall Success Rate (%)')
    ax.set_xscale('log')
    # Show all model sizes; offset 7B/8B labels to avoid overlap
    all_sizes = [3, 7, 8, 14, 32, 72]
    ax.set_xticks(all_sizes)
    ax.set_xticklabels(['3B', '7B', '8B', '14B', '32B', '72B'])
    ax.tick_params(axis='x', which='minor', bottom=False)
    # Nudge 7B/8B labels apart
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == '7B':
            lbl.set_ha('right')
        elif lbl.get_text() == '8B':
            lbl.set_ha('left')
    ax.set_ylim(0, 80)
    ax.legend(frameon=False, fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    
    fig.savefig('fig_scaling.pdf')
    fig.savefig('fig_scaling.png')
    plt.close(fig)
    print('Created fig_scaling.pdf')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    fig1_main_results()
    fig2_radar()
    fig3_ablations()
    fig4_scaling()
    print('All figures generated.')
