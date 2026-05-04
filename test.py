# 加载npz file，查看里面有什么内容
import numpy as np
data = np.load('/home/tmp_wanghf/armada/armada_data/maniskill_pick_rollout_poc/success_stats.npz')
print(data.files)
print(data['ot_values'])
print(data['ot_percentile'])