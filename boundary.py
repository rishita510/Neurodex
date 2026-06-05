import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df  = pd.read_csv("data12.csv")
ch1 = df['Channel1'].values
SR  = 2000

win  = int(SR * 0.1)
step = int(SR * 0.05)

rms = np.array([
    np.sqrt(np.mean(ch1[i:i+win]**2))
    for i in range(0, len(ch1)-win, step)
])
times = np.arange(len(rms)) * 0.05

# ── Auto detect REST zones ──────────────────────────────
threshold   = 0.004
min_rest    = 20
min_gesture = 40
is_rest     = rms < threshold

trans    = np.diff(is_rest.astype(int))
r_starts = times[np.where(trans ==  1)[0]]   # active → rest
r_ends   = times[np.where(trans == -1)[0]]   # rest → active

# Filter only real rest periods (>20s)
real_rests = [(rs, re) for rs, re in zip(r_starts, r_ends)
              if (re - rs) >= min_rest]

print("=" * 45)
print("DETECTED REST PERIODS:")
print("=" * 45)
for i, (rs, re) in enumerate(real_rests):
    print(f"  REST {i+1}: {rs:.0f}s → {re:.0f}s  (duration: {re-rs:.0f}s)")

print("\nSUGGESTED GESTURE BOUNDARIES:")
print("=" * 45)
g_starts = [0] + [re for rs, re in real_rests]
g_ends   = [rs for rs, re in real_rests] + [times[-1]]
gesture_zones = [(gs, ge) for gs, ge in zip(g_starts, g_ends)
                 if (ge - gs) >= min_gesture]
for i, (gs, ge) in enumerate(gesture_zones):
    print(f"  Gesture {i+1}: ({gs:.0f}, {ge:.0f})   duration={ge-gs:.0f}s")

# ── Plot ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 5))
ax.plot(times, rms, linewidth=0.8, color='steelblue', label='RMS')
ax.axhline(threshold, color='red', linestyle='--',
           linewidth=1, label=f'REST threshold ({threshold})')

# Find all points where RMS goes below threshold
below_threshold = times[rms < threshold]

# Mark on x-axis wherever signal goes below threshold
# Mark times where RMS is below threshold
for idx, t in enumerate(below_threshold):
    ax.axvline(t, color='orange', linewidth=0.3, alpha=0.3)

    # Show time label every 20th point to avoid clutter
    if idx % 20 == 0:
        ax.annotate(f'{t:.1f}s',
                    xy=(t, 0),
                    xycoords=('data', 'axes fraction'),
                    xytext=(0, -15),
                    textcoords='offset points',
                    rotation=90,
                    fontsize=7,
                    color='orange',
                    ha='center')

# Mark only the REAL rest period boundaries (>20s)
for i, (rs, re) in enumerate(real_rests):
    # Vertical lines at start and end of real rest
    ax.axvline(rs, color='red', linewidth=2, linestyle='-')
    ax.axvline(re, color='green', linewidth=2, linestyle='-')

    # Mark exact times ON the x-axis
    ax.annotate(f'{rs:.0f}s',
                xy=(rs, 0), xycoords=('data', 'axes fraction'),
                xytext=(0, -25), textcoords='offset points',
                color='red', fontsize=8, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='-', color='red', lw=1))

    ax.annotate(f'{re:.0f}s',
                xy=(re, 0), xycoords=('data', 'axes fraction'),
                xytext=(0, -25), textcoords='offset points',
                color='green', fontsize=8, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='-', color='green', lw=1))

# Label gesture zones
for i, (gs, ge) in enumerate(gesture_zones):
    mid = (gs + ge) / 2
    ax.text(mid, 0.002,
            f'G{i+1}\n{gs:.0f}→{ge:.0f}s',
            color='navy', fontsize=8, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("RMS")
ax.set_title("data1.csv — Times below threshold marked on X-axis")
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("boundaries.png", dpi=150, bbox_inches='tight')
plt.show()
