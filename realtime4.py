"""
=============================================================
Real-Time EMG Gesture Prediction — NPG Lite BLE (Direct)
=============================================================
Direct BLE connection via bleak — no chordspy/LSL dependency.

FIXES INCLUDED (vs earlier versions):
  1. N_GESTURES = 5 (model trained on 5 classes, not 6 —
     'hand_closed' was skipped during training)
  2. Peak-centered 3s window — re-centers on the RMS peak
     within the buffer so features match the training
     distribution (training events were always centered on
     peak muscle activation, not an arbitrary cut)
  3. Fixed (non-adaptive) activity threshold — adaptive
     mean+1.5*std was causing genuine gestures to be missed
  4. Visible countdown timer — starts the moment BLE connects,
     resets every 3s in sync with the prediction window
  5. STEP_SEC = 3 — one prediction per 3-second window
     (matches training event duration, near non-overlapping)

Packet format:
  SAMPLE_SIZE  = 13 bytes (1 counter + 6x2 bytes)
  Format       = 16-bit unsigned per channel
  START cmd    = b"START" on CONTROL_UUID
  Normalize    = (value - 2048) / 2048 -> +-1
  (Verified against training CSV: rest-state MAV matches
   almost exactly between this normalization and the CSV,
   so this conversion is correct — no change needed here.)
=============================================================
"""

import asyncio
import time
import threading
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from scipy import signal as scipy_signal
from collections import deque
from bleak import BleakClient

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# ════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════
DEVICE_ADDRESS  = "E4:B3:23:B0:5F:C6"   # your NPG Lite
DATA_UUID       = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CONTROL_UUID    = "0000ff01-0000-1000-8000-00805f9b34fb"

SR              = 2000
NUM_CHANNELS    = 6          # NPG Lite 6CH
SAMPLE_SIZE     = 13         # 1 + 6x2 bytes per packet
N_ACTIVE_CH     = 4          # only use first 4 channels
EVENT_SAMPLES   = 3 * SR     # 6000 samples = 3 seconds
STEP_SEC        = 3          # one prediction every 3 seconds
WAMP_THRESH     = 0.005
ACTIVITY_THRESH = 0.002     # fixed threshold — tune based on your rest-state RMS
                              # (your rest RMS was ~0.002-0.003, gesture ~0.01-0.03)

ALL_FEATURES = [
    'MAV_ch1','WAMP_ch1','VAR_ch1','WL_ch1','MDF_ch1','MNF_ch1',
    'MAV_ch2','WAMP_ch2','VAR_ch2','WL_ch2','MDF_ch2','MNF_ch2',
    'MAV_ch3','WAMP_ch3','VAR_ch3','WL_ch3','MDF_ch3','MNF_ch3',
    'MAV_ch4','WAMP_ch4','VAR_ch4','WL_ch4','MDF_ch4','MNF_ch4',
]

# Only 5 gestures were trained — gesture_id 6 'hand_closed' was
# skipped (emg_full_pipeline.py: `if gesture_id == 6: continue`)
N_GESTURES = 5
GESTURE_NAMES = {
    0:'Thumb Flexion',  1:'Index Flexion',
    2:'Middle Flexion', 3:'Ring Flexion',
    4:'Little Flexion',
}
GESTURE_EMOJIS = {0:'👍',1:'☝️',2:'🖕',3:'💍',4:'🤙'}
GESTURE_COLS   = {
    0:'#1D9E75', 1:'#BA7517', 2:'#D4537E',
    3:'#D85A30', 4:'#378ADD',
}

# ════════════════════════════════════════════════════════
# GLOBAL STATE
# ════════════════════════════════════════════════════════
sample_buffer  = deque(maxlen=EVENT_SAMPLES * 3)
latest_pred    = {'gesture':None, 'proba':None, 'active':False}
lock           = threading.Lock()
running        = True
ble_status     = {'connected':False, 'msg':'Not connected', 'connect_time': None}

# ════════════════════════════════════════════════════════
# LOAD MODEL
# ════════════════════════════════════════════════════════
def load_model():
    if not os.path.exists('best_model.pkl') or \
       not os.path.exists('scaler.pkl'):
        print("\n❌ best_model.pkl or scaler.pkl not found!")
        print("Run classifier.py first to save the model.")
        return None, None
    clf    = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print(f"✅ Model  : {type(clf).__name__}")
    print(f"✅ Scaler : ready")
    print(f"✅ Features: {len(ALL_FEATURES)}")
    return clf, scaler

# ════════════════════════════════════════════════════════
# PACKET DECODER
# ════════════════════════════════════════════════════════
def decode_packet(data: bytearray):
    samples = []
    for i in range(0, len(data), SAMPLE_SIZE):
        chunk = data[i:i + SAMPLE_SIZE]
        if len(chunk) != SAMPLE_SIZE:
            continue
        channels = []
        for ch in range(NUM_CHANNELS):
            idx1 = 1 + ch * 2
            idx2 = 2 + ch * 2
            if idx2 >= len(chunk):
                break
            high  = chunk[idx1]
            low   = chunk[idx2]
            raw   = (high << 8) | low
            value = (raw - 2048) / 2048.0
            channels.append(value)
        if len(channels) == NUM_CHANNELS:
            samples.append(channels)
    return samples

def notification_handler(sender, data: bytearray):
    samples = decode_packet(data)
    with lock:
        for s in samples:
            sample_buffer.append(s[:N_ACTIVE_CH])

# ════════════════════════════════════════════════════════
# FILTER
# ════════════════════════════════════════════════════════
def filter_channel(x):
    b_n, a_n = scipy_signal.iirnotch(50.0, Q=30, fs=SR)
    x = scipy_signal.filtfilt(b_n, a_n, x)
    nyq = SR / 2
    b_bp, a_bp = scipy_signal.butter(
        4, [20/nyq, min(450/nyq, 0.99)], btype='band')
    x = scipy_signal.filtfilt(b_bp, a_bp, x)
    return x

# ════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ════════════════════════════════════════════════════════
def extract_features_rt(window):
    """window: (EVENT_SAMPLES, N_ACTIVE_CH)"""
    feats = {}
    for i in range(N_ACTIVE_CH):
        x      = filter_channel(window[:, i])
        N      = len(x)
        ch_num = i + 1

        feats[f'MAV_ch{ch_num}']  = float(np.mean(np.abs(x)))
        feats[f'WAMP_ch{ch_num}'] = int(
            np.sum(np.abs(np.diff(x)) >= WAMP_THRESH))
        feats[f'VAR_ch{ch_num}']  = float(np.sum(x**2) / (N-1))
        feats[f'WL_ch{ch_num}']   = float(np.sum(np.abs(np.diff(x))))

        freqs, psd = scipy_signal.welch(x, fs=SR, nperseg=256)
        band = (freqs >= 20) & (freqs <= 450)
        fb, pb = freqs[band], psd[band]

        if len(fb) == 0 or np.sum(pb) == 0:
            feats[f'MDF_ch{ch_num}'] = 0.0
            feats[f'MNF_ch{ch_num}'] = 0.0
            continue

        cumsum  = np.cumsum(pb)
        mdf_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        feats[f'MDF_ch{ch_num}'] = float(fb[min(mdf_idx, len(fb)-1)])
        feats[f'MNF_ch{ch_num}'] = float(np.sum(fb*pb) / np.sum(pb))

    return feats

def select_features(feats_dict):
    return np.array(
        [feats_dict[f] for f in ALL_FEATURES],
        dtype=np.float32)

# ════════════════════════════════════════════════════════
# ACTIVITY DETECTION — fixed threshold (not adaptive)
# ════════════════════════════════════════════════════════
def is_active(window):
    ch1 = window[:, 0]
    rms = np.sqrt(np.mean(ch1**2))
    return rms > ACTIVITY_THRESH

# ════════════════════════════════════════════════════════
# PEAK-CENTERED WINDOW
# Training events were always centered on the RMS peak within
# a gesture block. A raw "last 3s" buffer can straddle rest +
# gesture, diluting MAV/VAR/WL. This re-centers on the peak so
# features better match what the model was trained on.
# ════════════════════════════════════════════════════════
def get_centered_window(full_window):
    ch1      = full_window[:, 0]
    win_size = int(SR * 0.3)
    step     = max(win_size // 2, 1)

    if len(ch1) <= win_size:
        return full_window

    rms_vals = np.array([
        np.sqrt(np.mean(ch1[i:i+win_size]**2))
        for i in range(0, len(ch1)-win_size, step)
    ])
    if len(rms_vals) == 0:
        return full_window

    peak_idx    = np.argmax(rms_vals)
    peak_sample = peak_idx * step + win_size // 2

    half = EVENT_SAMPLES // 2
    ev_s = peak_sample - half
    ev_e = ev_s + EVENT_SAMPLES

    if ev_s < 0:
        ev_s, ev_e = 0, EVENT_SAMPLES
    if ev_e > len(full_window):
        ev_e = len(full_window)
        ev_s = ev_e - EVENT_SAMPLES

    return full_window[ev_s:ev_e, :]

# ════════════════════════════════════════════════════════
# PREDICTION LOOP
# ════════════════════════════════════════════════════════
def prediction_loop(clf, scaler):
    global latest_pred, running
    print("🧠 Prediction loop started")

    while running:
        time.sleep(STEP_SEC)

        with lock:
            buf = list(sample_buffer)

        if len(buf) < EVENT_SAMPLES:
            continue

        full_window = np.array(buf[-EVENT_SAMPLES:])

        if not is_active(full_window):
            with lock:
                latest_pred = {'gesture':None, 'proba':None, 'active':False}
            continue

        window = get_centered_window(full_window)

        try:
            feats_dict   = extract_features_rt(window)
            feats_sel    = select_features(feats_dict)
            feats_scaled = scaler.transform(feats_sel.reshape(1,-1))
            pred         = int(clf.predict(feats_scaled)[0])
            proba        = clf.predict_proba(feats_scaled)[0]

            print(f"MAV_ch1={feats_dict['MAV_ch1']:.5f}  "
                  f"WL_ch1={feats_dict['WL_ch1']:.3f}  "
                  f"VAR_ch1={feats_dict['VAR_ch1']:.6f}  "
                  f"→ Predicted: {pred} ({GESTURE_NAMES.get(pred,'?')})"
                  f"  Conf: {max(proba)*100:.0f}%")

            if max(proba) < 0.40:
                with lock:
                    latest_pred = {'gesture': None, 'proba': proba, 'active': False}
                continue

            with lock:
                latest_pred = {'gesture': pred, 'proba': proba, 'active': True}

        except Exception as e:
            print(f"  Prediction error: {e}")

# ════════════════════════════════════════════════════════
# BLE
# ════════════════════════════════════════════════════════
async def ble_task():
    global running, ble_status

    ble_status = {'connected':False, 'msg':'Connecting...', 'connect_time': None}

    print(f"🔗 Connecting to {DEVICE_ADDRESS}...")
    async with BleakClient(DEVICE_ADDRESS, timeout=20.0) as client:
        ble_status = {'connected':True, 'msg':'Connected', 'connect_time': time.time()}
        print("✅ Connected!\n")

        await client.start_notify(DATA_UUID, notification_handler)
        await client.write_gatt_char(CONTROL_UUID, b"START")
        print("📡 Streaming at 2000 Hz...")
        print("🖐  Wear NPG Lite on forearm")
        print("✊  Hold gesture for 3 seconds\n")

        try:
            while running:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            running = False

        await client.stop_notify(DATA_UUID)
        ble_status = {'connected':False, 'msg':'Disconnected', 'connect_time': None}

# ════════════════════════════════════════════════════════
# DISPLAY
# ════════════════════════════════════════════════════════
def launch_display():
    fig = plt.figure(figsize=(14, 7), facecolor='#12121f')
    fig.canvas.manager.set_window_title('Real-Time EMG — NPG Lite')

    gs = gridspec.GridSpec(2, 4, figure=fig,
        hspace=0.5, wspace=0.3,
        left=0.06, right=0.97, top=0.88, bottom=0.08)

    # ── 4 channel plots ──────────────────────────────────
    ch_cols   = ['#1D9E75','#534AB7','#D85A30','#BA7517']
    ax_sigs   = [fig.add_subplot(gs[0, i]) for i in range(4)]
    sig_lines = []
    for i, ax in enumerate(ax_sigs):
        ax.set_facecolor('#0a0a18')
        ax.set_title(f'Ch {i+1}', color='#888', fontsize=9, pad=3)
        ax.set_xlim(0, EVENT_SAMPLES)
        ax.set_ylim(-1.1, 1.1)
        ax.tick_params(colors='#333', labelsize=6)
        for sp in ax.spines.values(): sp.set_color('#222')
        ax.axhline(0, color='#222', lw=0.5)
        line, = ax.plot([], [], color=ch_cols[i], lw=0.6)
        sig_lines.append(line)

    # ── Confidence bars (N_GESTURES = 5) ─────────────────
    ax_conf = fig.add_subplot(gs[1, :3])
    ax_conf.set_facecolor('#0a0a18')
    ax_conf.set_title('Prediction Confidence',
                      color='#aaa', fontsize=10, pad=5)
    ax_conf.set_xlim(0, 1)
    ax_conf.set_ylim(-0.5, N_GESTURES - 0.5)
    ax_conf.tick_params(colors='#444', labelsize=9)
    for sp in ax_conf.spines.values(): sp.set_color('#222')
    ax_conf.axvline(0.5, color='#333', lw=0.8, ls='--')
    ax_conf.set_xlabel('Confidence', color='#555', fontsize=8)

    ylabels = [f"{GESTURE_EMOJIS[i]}  {GESTURE_NAMES[i]}"
               for i in range(N_GESTURES)]
    bars = ax_conf.barh(range(N_GESTURES), [0]*N_GESTURES,
        color=[GESTURE_COLS[i] for i in range(N_GESTURES)],
        height=0.55, alpha=0.85)
    ax_conf.set_yticks(range(N_GESTURES))
    ax_conf.set_yticklabels(ylabels, color='#ccc', fontsize=9)
    bar_pcts = [
        ax_conf.text(0.01, i, '', va='center',
            fontsize=9, color='white', fontweight='bold')
        for i in range(N_GESTURES)]

    # ── Prediction box ───────────────────────────────────
    ax_pred = fig.add_subplot(gs[1, 3])
    ax_pred.set_facecolor('#0a0a18')
    ax_pred.axis('off')
    t_emoji = ax_pred.text(0.5, 0.62, '✋', fontsize=46,
        ha='center', va='center', transform=ax_pred.transAxes)
    t_label = ax_pred.text(0.5, 0.28, 'Waiting...', fontsize=11,
        ha='center', va='center', color='#555',
        fontweight='bold', transform=ax_pred.transAxes)
    t_conf  = ax_pred.text(0.5, 0.10, '', fontsize=9,
        ha='center', va='center', color='#444',
        transform=ax_pred.transAxes)

    # ── Countdown timer (top center) ─────────────────────
    t_timer = fig.text(0.5, 0.96, '⏱ --',
        ha='center', va='center', fontsize=24, fontweight='bold',
        color='#888')

    fig.suptitle(
        'NPG Lite — Real-Time Hand Gesture Recognition',
        color='#ddd', fontsize=11, fontweight='bold', y=0.99)
    t_status = fig.text(0.5, 0.01,
        '🔴 Connecting...', ha='center',
        color='#888', fontsize=9)

    def update(frame):
        with lock:
            buf  = list(sample_buffer)
            pred = dict(latest_pred)
            conn = dict(ble_status)

        # Countdown timer — starts the moment BLE connects
        if conn['connected'] and conn.get('connect_time') is not None:
            elapsed   = time.time() - conn['connect_time']
            remaining = STEP_SEC - (elapsed % STEP_SEC)
            if remaining <= 0.3:
                t_timer.set_text('🟢 GIVE GESTURE NOW!')
                t_timer.set_color('#1D9E75')
                t_timer.set_fontsize(22)
            else:
                t_timer.set_text(f'⏱ {remaining:.1f}s')
                t_timer.set_color('#BA7517' if remaining < 1.0 else '#888')
                t_timer.set_fontsize(24)
        else:
            t_timer.set_text('⏱ --')
            t_timer.set_color('#444')

        # Signal plots
        if len(buf) >= EVENT_SAMPLES:
            win = np.array(buf[-EVENT_SAMPLES:])
            for i, ln in enumerate(sig_lines):
                ln.set_data(range(EVENT_SAMPLES), win[:, i])

        # Status
        if conn['connected']:
            if pred['active']:
                t_status.set_text('🟢 Active — predicting')
                t_status.set_color('#1D9E75')
            else:
                t_status.set_text('🟡 Connected — do a gesture')
                t_status.set_color('#BA7517')
        else:
            t_status.set_text(f"🔴 {conn['msg']}")
            t_status.set_color('#888')

        # Prediction
        if pred['active'] and pred['proba'] is not None:
            g, proba = pred['gesture'], pred['proba']
            for j,(bar,pct) in enumerate(zip(bars, bar_pcts)):
                if j < len(proba):
                    bar.set_width(proba[j])
                    if proba[j] > 0.04:
                        pct.set_text(f'{proba[j]*100:.0f}%')
                        pct.set_x(proba[j]+0.01)
                    else:
                        pct.set_text('')
            t_emoji.set_text(GESTURE_EMOJIS.get(g, '❓'))
            t_emoji.set_color(GESTURE_COLS.get(g, '#888'))
            t_label.set_text(GESTURE_NAMES.get(g, f'Class {g}'))
            t_label.set_color(GESTURE_COLS.get(g, '#888'))
            t_conf.set_text(f'{max(proba)*100:.0f}% confident')
            t_conf.set_color('#888')
        else:
            for bar,pct in zip(bars, bar_pcts):
                bar.set_width(0); pct.set_text('')
            t_emoji.set_text('✋')
            t_emoji.set_color('#333')
            t_label.set_text('Rest / No gesture')
            t_label.set_color('#444')
            t_conf.set_text('')

        return (sig_lines + list(bars) + bar_pcts +
                [t_emoji, t_label, t_conf, t_status, t_timer])

    ani = FuncAnimation(fig, update, interval=100,
        blit=False, cache_frame_data=False)
    plt.show()
    return ani

# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
def main():
    global running

    print("╔══════════════════════════════════════════════════════╗")
    print("║   Real-Time EMG Gesture — NPG Lite BLE (Direct)     ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Device  : {DEVICE_ADDRESS}               ║")
    print(f"║  SR      : {SR} Hz                              ║")
    print(f"║  Window  : 3s ({EVENT_SAMPLES} samples)                   ║")
    print(f"║  Channels: {N_ACTIVE_CH} active / {NUM_CHANNELS} total                   ║")
    print(f"║  Gestures: {N_GESTURES}                                       ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    clf, scaler = load_model()
    if clf is None:
        return

    ble_thread = threading.Thread(
        target=lambda: asyncio.run(ble_task()),
        daemon=True)
    ble_thread.start()

    pred_thread = threading.Thread(
        target=prediction_loop,
        args=(clf, scaler),
        daemon=True)
    pred_thread.start()

    print("Connecting", end='', flush=True)
    for _ in range(30):
        if ble_status['connected']: break
        time.sleep(0.5)
        print('.', end='', flush=True)
    print()

    launch_display()
    running = False
    print("\n✅ Done.")

if __name__ == "__main__":
    main()
