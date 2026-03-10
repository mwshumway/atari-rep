# atari-rep

## SPR baseline compatibility mode

To run online RL with settings aligned to the SPR DQN baseline (without using `rlpyt`), enable:

`--agent.spr_baseline_compat`

Example:

`python run_online_rl.py --games seaquest --agent.spr_baseline_compat --agent.num_timesteps 100000 --agent.rollout_freq 10000`

This mode applies a Nature CNN + Identity neck setup and SPR-like optimizer/replay/scheduling defaults for baseline comparability.

## Baseline gap ablation (legacy vs SPR-compat)

Launch a matched ablation (same seeds/games, two variants):

`python scripts/online_rl/baseline_gap_ablation.py --action launch --mode qsub --games seaquest --seeds 0 1 2 --project spr_baseline_gap --group-prefix gap_ablation`

By default launch mode does not force a WandB entity (it uses your local login/default). If needed, pass `--entity <your_entity>`.

Preview generated run commands without launching:

`python scripts/online_rl/baseline_gap_ablation.py --action launch --mode print`

After runs finish, summarize the gap from WandB summaries:

`python scripts/online_rl/baseline_gap_ablation.py --action summarize --project spr_baseline_gap --entity mshumway-boston-university --group-prefix gap_ablation --metric eval/mean_traj_game_scores`

You can also summarize normalized score instead of raw game score with `--metric eval/dns`.
