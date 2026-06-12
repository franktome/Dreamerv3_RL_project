#!/usr/bin/env python3
"""Subprocess worker: DreamerV2 Atari inference (isolates TF from JAX parent)."""

import json
import pathlib
import sys

import numpy as np


def _action_entropy(actions: list[int]) -> float:
    if not actions:
        return 0.0
    counts = np.bincount(np.asarray(actions, dtype=np.int64))
    p = counts[counts > 0] / len(actions)
    return float(-(p * np.log(p + 1e-12)).sum())


def main():
    payload = json.loads(sys.argv[1])
    root = pathlib.Path(payload['dreamerv2_root'])
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'dreamerv2'))

    import tensorflow as tf

    tf.config.experimental_run_functions_eagerly(not False)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    import agent
    import common

    logdir = pathlib.Path(payload['logdir'])
    config = common.Config.load(logdir / 'config.yaml')
    config = config.update({'jit': False, 'precision': 32})
    overrides = payload.get('env_overrides') or {}
    game = payload['game']
    action_repeat = int(overrides.get('repeat', config.action_repeat))
    sticky = overrides.get('sticky', 0.25)
    sticky = float(sticky) if not isinstance(sticky, bool) else (0.25 if sticky else 0.0)
    noops = int(overrides.get('noops', 30))

    env = common.Atari(
        game, action_repeat, config.render_size, config.atari_grayscale,
        noops=noops, sticky=sticky)
    env = common.OneHotAction(env)
    env = common.TimeLimit(env, config.time_limit)

    step = common.Counter()
    agnt = agent.Agent(config, env.obs_space, env.act_space, step)
    replay = common.Replay(logdir / 'train_episodes', **config.replay)
    train_agent = common.CarryOverState(agnt.train)
    train_agent(next(iter(replay.dataset(**config.dataset))))
    variables = payload.get('variables') or (logdir / 'variables.pkl')
    agnt.load(pathlib.Path(variables))
    policy = lambda *args, **kwargs: agnt.policy(*args, mode='eval', **kwargs)

    frames, actions_log, total_reward = [], [], 0.0
    max_steps = int(payload['max_steps'])

    def on_step(tran, worker):
        nonlocal total_reward
        if tran.get('is_first', False):
            return
        total_reward += float(tran.get('reward', 0))
        if len(frames) * 3 < max_steps or len(frames) < max_steps // 3:
            img = np.asarray(tran['image'])
            if img.ndim == 3 and img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            # DreamerV2 Atari env rotates 90° CCW vs DreamerV3; undo for display GIFs.
            img = np.rot90(img.astype(np.uint8), k=-1)
            frames.append(img.copy())
        if 'action' in tran:
            act = np.asarray(tran['action'])
            actions_log.append(int(np.argmax(act)))

    driver = common.Driver([env])
    driver.on_step(on_step)
    driver(policy, steps=max_steps)

    import imageio

    out_gif = pathlib.Path(payload['out_gif'])
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        imageio.mimsave(str(out_gif), frames[: max_steps // 3 + 1], duration=125)
    print(json.dumps({
        'ok': True, 'gif': str(out_gif), 'score': round(total_reward, 3),
        'steps': len(actions_log), 'action_entropy': _action_entropy(actions_log),
    }))


if __name__ == '__main__':
    main()
