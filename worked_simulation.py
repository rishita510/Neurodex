import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("newme.xml")
data = mujoco.MjData(model)

jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "thumb_cmc")
qpos_adr = model.jnt_qposadr[jid]
qvel_adr = model.jnt_dofadr[jid]

act_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_thumb_cmc_x")
act_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_thumb_cmc_y")
act_z = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_thumb_cmc_z")

act_thumb_pip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_thumb_pip_flex")
act_thumb_dip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_thumb_dip_flex")
act_index_mcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_index_mcp_flex")
act_index_pip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_index_pip_flex")
act_index_dip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_index_dip_flex")
act_ring_mcp  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ring_mcp_flex")
act_ring_pip  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ring_pip_flex")
act_ring_dip  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ring_dip_flex")
act_pinky_mcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_pinky_mcp_flex")
act_pinky_pip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_pinky_pip_flex")
act_pinky_dip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_pinky_dip_flex")

KP = 6.0
KD = 0.4

def quat_to_axis_angle(quat):
    w, x, y, z = quat
    w = np.clip(w, -1.0, 1.0)
    angle = 2 * np.arccos(w)
    s = np.sqrt(max(1 - w * w, 1e-12))
    if s < 1e-8:
        return np.zeros(3)
    return np.array([x, y, z]) / s * angle

# ── pause state ────────────────────────────────────────────────────────────────
paused = False

def key_callback(keycode):
    global paused
    SPACE = 32          # ASCII / GLFW key code for spacebar
    if keycode == SPACE:
        paused = not paused
        print("PAUSED" if paused else "RESUMED")
# ──────────────────────────────────────────────────────────────────────────────

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:

    t = 0.0

    while viewer.is_running():

        if paused:
            viewer.sync()          # keep the window alive and responsive
            time.sleep(0.016)      # ~60 fps idle
            continue               # skip physics + ctrl writes entirely

        # --- thumb CMC ball-joint PD control ---
        swing = 0.4 * (0.5 + 0.5 * np.sin(t))
        target_euler = np.array([swing, 0.2 * np.sin(t * 0.5), 0.0])
        target_quat = np.zeros(4)
        mujoco.mju_euler2Quat(target_quat, target_euler, "xyz")

        current_quat = data.qpos[qpos_adr:qpos_adr + 4]
        quat_err = np.zeros(4)
        mujoco.mju_mulQuat(quat_err, target_quat,
                           np.array([current_quat[0], -current_quat[1],
                                     -current_quat[2], -current_quat[3]]))
        err_vec = quat_to_axis_angle(quat_err)
        ang_vel = data.qvel[qvel_adr:qvel_adr + 3]
        torque = KP * err_vec - KD * ang_vel
        data.ctrl[act_x] = np.clip(torque[0], -2, 2)
        data.ctrl[act_y] = np.clip(torque[1], -2, 2)
        data.ctrl[act_z] = np.clip(torque[2], -2, 2)

        # --- middle finger ---
        curl = 0.8 * (0.5 + 0.5 * np.sin(t))
        data.ctrl[0] = curl
        data.ctrl[1] = curl * 1.1
        data.ctrl[2] = curl * 0.8

        # --- index finger ---
        index_curl = 0.8 * (0.5 + 0.5 * np.sin(t))
        data.ctrl[act_index_mcp] = index_curl
        data.ctrl[act_index_pip] = index_curl * 1.1
        data.ctrl[act_index_dip] = index_curl * 0.8

        # --- thumb PIP / DIP ---
        thumb_curl = 0.55 * (0.5 + 0.5 * np.sin(t))
        data.ctrl[act_thumb_pip] = thumb_curl * 1.0
        data.ctrl[act_thumb_dip] = thumb_curl * 0.85

        # --- ring finger ---
        ring_curl = 0.8 * (0.5 + 0.5 * np.sin(t))
        data.ctrl[act_ring_mcp] = ring_curl
        data.ctrl[act_ring_pip] = ring_curl * 1.1
        data.ctrl[act_ring_dip] = ring_curl * 0.8

        # --- pinky finger ---
        pinky_curl = 0.8 * (0.5 + 0.5 * np.sin(t))
        data.ctrl[act_pinky_mcp] = pinky_curl
        data.ctrl[act_pinky_pip] = pinky_curl * 1.1
        data.ctrl[act_pinky_dip] = pinky_curl * 0.8

        mujoco.mj_step(model, data)
        viewer.sync()

        t += 0.02
        time.sleep(0.002)