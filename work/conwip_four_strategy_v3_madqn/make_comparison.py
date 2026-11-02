"""Render the four-policy comparison for the frozen (3,3,2) PairV2 setup."""
import argparse,csv,json
from pathlib import Path

DISPLAY={
    'fixed_conwip':'Fixed CONWIP (training-seed scan)',
    'coordinated_pi_reconstructed':'Coordinated PI (reconstructed)',
    'madqn':'MADQN (trained in this experiment)',
    'dqn':'Centralized DQN (existing matched-seed training)',
}

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='results/four_strategy_v2_madqn')
    p.add_argument('--csv',default='comparison_four_madqn.csv');p.add_argument('--md',default='Four_Strategies_MADQN.md')
    a=p.parse_args();root=Path(a.results);summary=json.loads((root/'comparison_new_environment.json').read_text(encoding='utf-8'))
    expected={'fixed_conwip','coordinated_pi_reconstructed','madqn','dqn'}
    if {r['method'] for r in summary}!=expected:raise ValueError('Expected exactly the four configured strategies')
    order={'fixed_conwip':0,'coordinated_pi_reconstructed':1,'madqn':2,'dqn':3}
    summary.sort(key=lambda r:order[r['method']])
    rows=[]
    for s in summary:
        rows.append(dict(method=s['method'],display=DISPLAY[s['method']],n=s['n'],
            reward_mean=s['reward_x1000_mean'],reward_sd=s['reward_x1000_sd'],
            nrmse_mean=s['nrmse_mean'],nrmse_sd=s['nrmse_sd'],
            mean_wip=s['mean_wip_mean'],mean_wip_sd=s['mean_wip_sd'],
            cap_adjustment_mean=s['cap_adjustment_points_mean'],cap_adjustment_sd=s['cap_adjustment_points_sd'],
            environment='PairV2 environment (3,3,2)'))
    with Path(a.csv).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# PairV2 Four-Strategy Comparison (MADQN Variant)','',
      'The frozen environment uses initial caps (3,3,2), 720-minute warm-up and 3600-minute control, unchanged targets/disturbances, WIP weight 0.05 and action-change weight 0.005. Policies control only three integer cell WIP caps.','',
      '| Strategy | n | Reward ×1000 | Normalized RMSE | Mean total WIP | Cap-change decisions |','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['display']} | {r['n']} | {r['reward_mean']:.3f} ± {r['reward_sd']:.3f} | {r['nrmse_mean']:.3f} ± {r['nrmse_sd']:.3f} | {r['mean_wip']:.3f} ± {r['mean_wip_sd']:.3f} | {r['cap_adjustment_mean']:.3f} ± {r['cap_adjustment_sd']:.3f} |")
    lines += ['', 'Fixed-CONWIP scans and central-DQN training reuse matched-environment/seed results. New MADQN uses training seeds 13101–13105, 300 episodes each, and evaluation seeds 14101–14112. PI uses frozen parameters but was reconstructed because its equations were absent from the source archive.',
      '', "MADQN comprises three independent local Double-DQNs. Each cell reads its own 45-D observation and selects decrease/hold/increase. Cells share team reward and joint replay. Evaluation deterministically selects each agent's highest-Q legal action.",
      '', 'This is a synthetic SimPy research environment, not an AnyLogic numerical replica or a calibrated real factory. NRMSE≤0.10 must be established by measured test results.', '']
    Path(a.md).write_text('\n'.join(lines),encoding='utf-8')
    print('Four-strategy MADQN comparison files are ready.')

if __name__=='__main__':main()
