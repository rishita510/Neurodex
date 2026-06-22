"""
=============================================================
BLE Session Recorder — cued protocol, 5 gestures
=============================================================
Per gesture: 20 reps x 3s (60s active) + 30s rest = 90s
Order: Thumb, Index, Middle, Ring, Little (edit GESTURES below)
Total: 450s (7.5 min)

Prints live cues so you know exactly when to perform each rep.
Records RAW samples with arrival timestamps, measures the TRUE
sample rate from those timestamps, then filters using that real
rate (not a hardcoded assumption).

Usage:
  python session_recorder.py session1.csv
=============================================================
"""

import asyncio
import sys
import time
import pandas as pd
from scipy import signal
from bleak import BleakClient

DEVICE_ADDRESS = "E4:B3:23:B0:5F:C6"
DATA_UUID      = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CONTROL_UUID   = "0000ff01-0000-1000-8000-00805f9b34fb"

SAMPLE_SIZE  = 13
NUM_CHANNELS = 6

GESTURES   = ["Thumb", "Index", "Middle", "Ring", "Little"]
REPS       = 20
REP_SEC    = 3
REST_SEC   = 30
ACTIVE_SEC = REPS * REP_SEC   # 60s

rows       = []
timestamps = []   # (perf_counter_time, n_samples_in_packet)
counter    = 0


def decode_packet(data: bytearray):
    samples = []
    for i in range(0, len(data), SAMPLE_SIZE):
        chunk = data[i:i + SAMPLE_SIZE]
        if len(chunk) != SAMPLE_SIZE:
            continue
        channels = []
        for ch in range(NUM_CHANNELS):
            idx1, idx2 = 1 + ch*2, 2 + ch*2
            if idx2 >= len(chunk):
                break
            raw = (chunk[idx1] << 8) | chunk[idx2]
            channels.append((raw - 2048) / 2048.0)
        if len(channels) == NUM_CHANNELS:
            samples.append(channels)
    return samples


def notification_handler(sender, data: bytearray):
    global counter
    now = time.perf_counter()
    decoded = decode_packet(data)
    for s in decoded:
        rows.append([counter] + s)
        counter += 1
    timestamps.append((now, len(decoded)))


def filter_channel(x, sr):
    b_n, a_n = signal.iirnotch(50.0, Q=30, fs=sr)
    x = signal.filtfilt(b_n, a_n, x)
    nyq = sr / 2
    high = min(450 / nyq, 0.99)
    b_bp, a_bp = signal.butter(4, [20/nyq, high], btype='band')
    x = signal.filtfilt(b_bp, a_bp, x)
    return x


async def run_protocol():
    """Prints live cues for the full 5-gesture cued protocol."""
    session_t0 = time.time()

    for g_idx, gesture in enumerate(GESTURES):
        block_start = time.time()
        print(f"\n{'='*55}")
        print(f"GESTURE {g_idx+1}/5: {gesture.upper()}")
        print(f"{'='*55}")

        for rep in range(1, REPS + 1):
            rep_t0 = time.time()
            print(f"  Rep {rep:2d}/20  →  PERFORM {gesture.upper()} NOW", flush=True)
            # wait out the 3s rep window, printing nothing else (avoid spam)
            while time.time() - rep_t0 < REP_SEC:
                await asyncio.sleep(0.05)

        print(f"\n  ✅ {gesture} done. REST {REST_SEC}s ...")
        rest_t0 = time.time()
        while time.time() - rest_t0 < REST_SEC:
            remaining = REST_SEC - (time.time() - rest_t0)
            print(f"\r  💤 Resting... {remaining:4.1f}s left", end="", flush=True)
            await asyncio.sleep(0.1)
        print()

    print(f"\n✅ Protocol complete. Total elapsed: {time.time()-session_t0:.1f}s")


async def main(out_path: str):
    print(f"Connecting to {DEVICE_ADDRESS} ...")
    async with BleakClient(DEVICE_ADDRESS, timeout=20.0) as client:
        print("✅ Connected.")
        await client.start_notify(DATA_UUID, notification_handler)
        await client.write_gatt_char(CONTROL_UUID, b"START")
        await asyncio.sleep(1.0)   # let stream stabilize before protocol starts

        await run_protocol()

        await client.stop_notify(DATA_UUID)

    print(f"\nRecorded {len(rows)} samples. Computing real sample rate...")

    if len(timestamps) >= 2:
        t_first, n_first   = timestamps[0]
        t_last              = timestamps[-1][0]
        total_after_first   = sum(n for _, n in timestamps[1:])
        span = t_last - t_first
        measured_sr = total_after_first / span if span > 0 else 500.0
    else:
        measured_sr = 500.0

    print(f"📏 Measured SR: {measured_sr:.1f} Hz  ← USE THIS in your pipeline's SR constant")

    df = pd.DataFrame(rows, columns=["Counter"] + [f"Channel{i+1}" for i in range(NUM_CHANNELS)])
    for ch in [f"Channel{i+1}" for i in range(4)]:
        df[ch] = filter_channel(df[ch].values, sr=measured_sr)

    df.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")
    print(f"\nGesture order recorded: {GESTURES}")
    print(f"Each block: {ACTIVE_SEC}s active + {REST_SEC}s rest = {ACTIVE_SEC+REST_SEC}s")
    print(f"⚠ Use measured_sr={measured_sr:.1f} for segmentation/filtering downstream, not 2000.")


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "session.csv"
    asyncio.run(main(out_file))
