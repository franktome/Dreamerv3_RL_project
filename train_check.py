import json
import matplotlib.pyplot as plt

# 데이터 로드
with open('logdir/highway_train/scores.jsonl') as f:
    data = [json.loads(l) for l in f]

steps = [d['step'] for d in data if 'episode/score' in d]
scores = [d['episode/score'] for d in data if 'episode/score' in d]

# 이동 평균
window = 20
avg = [sum(scores[max(0,i-window):i+1]) / min(i+1,window)
       for i in range(len(scores))]

plt.figure(figsize=(10, 4))
plt.plot(steps, scores, alpha=0.3, label='raw')
plt.plot(steps, avg, label=f'moving avg ({window})')
plt.xlabel('Steps')
plt.ylabel('Episode Score')
plt.title('DreamerV3 on highway-fast-v0')
plt.legend()
plt.tight_layout()
plt.savefig('highway_train_curve.png', dpi=150)
plt.show()