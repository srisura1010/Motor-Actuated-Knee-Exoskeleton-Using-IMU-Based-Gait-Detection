import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter

#NOTE: τ is global symbol for torque it will be referred to as tau

# Global styles for Graphs in MATPLOTLIB
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    14,
    'axes.titleweight':  'bold',
    'axes.labelsize':    12,
    'axes.labelweight':  'bold',
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'legend.framealpha': 0.9,
    'grid.alpha':        0.35,
    'grid.linestyle':    '--',
    'figure.dpi':        150,
    'savefig.dpi':       200,
    'savefig.bbox':      'tight',
})

#global colors
CC  = '#2E86AB' #steel blue
CU  = '#E07B39' #orange
CBK = '#1A1A2E' #near-black
CGR = '#27AE60' #green

FIGSIZE = (10, 5.5)

def style_ax(ax, xlabel, ylabel, title, xlim=(0, 100)):
    ax.set_xlabel(xlabel) #x axis definition
    ax.set_ylabel(ylabel) #y axis definition
    ax.set_title(title, pad=10)
    ax.set_xlim(xlim)
    ax.grid(True)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    #the rest is just defaults for style of graph to be constant

def add_window(ax, label=True):
    ax.axvspan(30, 60, alpha=0.08, color=CGR,
               label='Assist Window (30–60%)' if label else None)
#highlights key 30-60% output

# 1. DATA LOADING & KINEMATICS ___________________________________________________
mean_curve = np.loadtxt("normative_mean_curve.csv", delimiter=",")
std_curve  = np.loadtxt("normative_std_curve.csv",  delimiter=",")
#loads the plain CV file into numpy array and delimiter separates columns by comma

mean_curve_original=mean_curve.copy()
mean_curve = -mean_curve
#CMU data stores flexion as negative so convert to standard convention

gait_percent = np.linspace(0, 100, len(mean_curve)) #evenly spaced data points for plotting
T  = 1.2
t  = (gait_percent / 100) * T
dt = t[1] - t[0]
# gait cycle period in seconds and separation of seconds into samples

theta_raw = np.deg2rad(mean_curve) #converts to radians
theta     = savgol_filter(theta_raw, window_length=15, polyorder=4)
omega_raw = np.gradient(theta, dt) #derivative d theta/dt which gives angular velocity
omega     = savgol_filter(omega_raw, window_length=15, polyorder=4)
alpha     = np.gradient(omega, dt) #derivative of omega gives angular acceleration -> acceleration is derivative of velocity
# Savitzky-Golay prevents noise and smooths graph on double differentiation

# 2. PHYSICAL MODEL ________________________________________________________________

I     = 0.05 # kg·m² -> moment of inertia (shank + brace)
m     = 4.0 # kg -> shank + brace mass
L     = 0.2 #m -> distance from knee to segment center of mass
b     = 0.1 #Nm·s/rad joint damping term
g_val = 9.81 #m/s² gravitational constant

#Newton–Euler: τ = Iα + bω + mgL·sin(θ) -> formula for torque
tau_human = I * alpha + b * omega + m * g_val * L * np.sin(theta)

# 3. HARDWARE PARAMS ___________________________________________________________

Kt            = 0.094 #Nm/A - motor torque constant from RO60 KV115 Specs
I_cont        = 8.5 #A continuous current limit (more will overheat motor windings)
V_CUR, Ah_CUR = 24.0, 5.0 #battery limitations
V_UPG, Ah_UPG = 22.2, 5.0 #upgraded battery limitations
GEAR_UPG      = 30.0 #cycloidal gearbox ratio
EFF_UPG       = 0.85    #average cycloidal gearbox efficiency

# tau_joint = Kt × I_cont × GearRatio × Efficiency
TAU_JOINT_CUR = Kt * I_cont * 1.0 #hardware ceiling, current
TAU_JOINT_UPG = Kt * I_cont * GEAR_UPG * EFF_UPG #upgraded ceiling

Kp = 3.0;  Ki = 0.8;  Kd = 0.05 #values used in PID tuning (plugged in different values and tested to find values that worked)
ASSIST_RATIO  = 0.40 #request 40% of human torque to provide 40% workload decrease target
TORQUE_THRESH = 1.0 #Nm - minimum human torque before assist activates

mask = (gait_percent >= 30) & (gait_percent <= 60) #assist window mask

# 4. CONTROLLER ___________________________________________________________________
def desired_assist(tau_h, gait_pct):
    assist = np.zeros_like(tau_h) #we don't want to assist outside window, this ensures it

    for idx in range(len(tau_h)): #30-60% and if human torque above 1Nm threshold
        if 30 <= gait_pct[idx] <= 60 and tau_h[idx] > TORQUE_THRESH:
            assist[idx] = ASSIST_RATIO * tau_h[idx]

    #Smooth desired torque —> reduces high-frequency PID hunting
    #Savgol is more simple than Kalman filtering and effective enough for this simulation
    #without a filter PID cannot tune because of sharp edge -> causes overshoot
    assist = savgol_filter(assist, window_length=11, polyorder=3)
    assist = np.clip(assist, 0, None)   # don't let smoothing create negatives
    return assist

def pid(tau_des, tau_lim, dt):
    #u(t) = Kp·e + Ki·∫e dt + Kd·de/dt   with anti-windup + hard clamp
    out = np.zeros_like(tau_des)
    integral = 0.0
    prev_e   = 0.0
    for i in range(len(tau_des)):
        p  = out[i-1] if i > 0 else 0.0
        e  = tau_des[i] - p
        if p < tau_lim:
            integral += e * dt #how far you are off
        d  = (e - prev_e) / dt if i > 0 else 0.0
        u  = Kp*e + Ki*integral + Kd*d
        u  = max(0.0, min(u, tau_lim))   # clamp: never oppose user, never exceed limit
        out[i]  = u
        prev_e  = e
    return out

#p is reacting to error, big error = push harder
#i is reacting to the error accumulated over time
#d term reacts to how fast the error is changing, if error is shrinking rapidly, ease off


# 5. CURRENT HARDWARE SIMULATION  (primary result)______________________________________

tau_des_cur    = desired_assist(tau_human, gait_percent) #compute what motor needs to do
tau_motor_cur  = pid(tau_des_cur, TAU_JOINT_CUR, dt) #run the pid control loop
tau_motor_cur  = np.where(tau_human > 0, tau_motor_cur, 0.0) #Where tau_human > 0 is True, keep tau_motor_cur; where it's False, replace with 0.0.
tau_assist_cur = tau_human - tau_motor_cur #remaining human effort metric

# mechanical power = torque times angular velocity so integrate that power over time using ->
# trapezoidal sum (opposed to reimann) to get total energy used per cycle in joules
power_cur      = np.abs(tau_motor_cur * omega)
energy_cur     = np.trapezoid(power_cur, t)
avg_power_cur  = energy_cur / T

#total mechanical work knee does in one gait cycle without assistance
human_work     = np.trapezoid(np.abs(tau_human * omega), t)
work_red_cur   = 100 * (human_work - np.trapezoid(np.abs(tau_assist_cur * omega), t)) / human_work
I_draw_cur     = tau_motor_cur / Kt   # I = τ / Kt  (no gearbox)

#Rearranging τ = Kt × I → I = τ / Kt. With no gearbox, the joint torque equals the motor shaft torque ->
#so this directly gives motor current in amps.
pct_assist_cur = np.where(
    (np.abs(tau_human) > 2.5) & mask,
    np.clip(100 * (tau_motor_cur / np.abs(tau_human)), 0, 100), 0)

rt_cur = (V_CUR * Ah_CUR) / (avg_power_cur + 5.5) #runtime estimate

# 6. UPGRADED SIMULATION  (engineering analysis / future work) _________________________________________________________
# PID ceiling = 40% × peak human torque = 5.12 Nm
# TAU_JOINT_UPG = 20.4 Nm > 5.12 Nm
# Motor shaft current = τ_joint / GearRatio / Efficiency / Kt

# keep in mind that much of the logic behind upgraded is same as original, values change however

tau_des_upg    = desired_assist(tau_human, gait_percent) #same desired torque
pid_clamp_upg  = ASSIST_RATIO * tau_human.max() #PID should not ask for more than 40% (assist ratio) of what human needs
tau_motor_upg  = pid(tau_des_upg, pid_clamp_upg, dt)  #runs the pid loop with new ceiling
tau_motor_upg  = np.where(tau_human > 0, tau_motor_upg, 0.0) #same logic
tau_assist_upg = tau_human - tau_motor_upg #remaining human effort after motor assists

power_upg      = np.abs(tau_motor_upg * omega) #same power and energy calc
energy_upg     = np.trapezoid(power_upg, t)
avg_power_upg  = energy_upg / T
work_red_upg   = 100 * (human_work - np.trapezoid(np.abs(tau_assist_upg * omega), t)) / human_work
#percentage of human mechanical work that the exo takes over, same formula with upgraded torques
I_draw_upg     = (tau_motor_upg / GEAR_UPG / EFF_UPG) / Kt
#To find motor current you have to work backwards through the gearbox — divide joint torque by gear ratio and ->
#efficiency to get motor shaft torque, then divide by Kt to get current

pct_assist_upg = np.where(
    (np.abs(tau_human) > 2.5) & mask, #2.5 avoids division by 0 error so its a good check
    np.clip(100 * (tau_motor_upg / np.abs(tau_human)), 0, 100), 0)

rt_upg = (V_UPG * Ah_UPG) / (avg_power_upg + 5.5) #runtime formula

# Smoothed versions for display graphs -> physics/energy calcs use raw above
tau_motor_cur_disp  = savgol_filter(tau_motor_cur,  window_length=9, polyorder=3)
tau_motor_upg_disp  = savgol_filter(tau_motor_upg,  window_length=9, polyorder=3)

#smooths the other data out for display as well so graphs don't show jagged lines and instead smooth curves for clean current plots
tau_assist_cur_disp = tau_human - tau_motor_cur_disp
tau_assist_upg_disp = tau_human - tau_motor_upg_disp
I_draw_cur_disp     = tau_motor_cur_disp / Kt
I_draw_upg_disp     = (tau_motor_upg_disp / GEAR_UPG / EFF_UPG) / Kt


# 7. PRINT RESULTS __________________________________________________________________________________________________

print("\n" + "="*56)
print("  CURRENT HARDWARE — PRIMARY RESULTS")
print("="*56)
print(f"  Work Reduction:        {work_red_cur:.2f}%")
print(f"  Motor Energy/cycle:    {energy_cur:.3f} J")
print(f"  Avg Motor Power:       {avg_power_cur:.2f} W")
print(f"  Avg Assist (30–60%):   {np.mean(pct_assist_cur[mask]):.2f}%")
print(f"  Peak Current:          {I_draw_cur.max():.2f} A  (limit: {I_cont} A)")
print(f"  Avg Current (active):  {np.mean(I_draw_cur[I_draw_cur>0]):.2f} A")
print(f"  System Runtime:        {rt_cur:.1f} h")
print(f"  Knee angle range:      {np.rad2deg(theta).min():.1f}° to {np.rad2deg(theta).max():.1f}°")
print(f"  tau_human range:       {tau_human.min():.2f} to {tau_human.max():.2f} Nm")
print(f"  tau_motor_cur max:     {tau_motor_cur.max():.3f} Nm  (ceiling: {TAU_JOINT_CUR:.3f} Nm)")

print("\n" + "="*56)
print("  UPGRADED (30:1) — ENGINEERING ANALYSIS")
print("="*56)
print(f"  Work Reduction:        {work_red_upg:.2f}%")
print(f"  Avg Assist (30–60%):   {np.mean(pct_assist_upg[mask]):.2f}%")
print(f"  Peak Current:          {I_draw_upg.max():.2f} A  (limit: {I_cont} A)")
print(f"  System Runtime:        {rt_upg:.1f} h")
print(f"  tau_motor_upg max:     {tau_motor_upg.max():.3f} Nm  (clamp: {pid_clamp_upg:.3f} Nm)")

print("\n" + "="*56)
print("  DELTA (Upgraded vs Current)")
print("="*56)
print(f"  Work Reduction Gain:   {work_red_upg - work_red_cur:+.2f}%")
print(f"  Motor Current Change:  {I_draw_upg.max() - I_draw_cur.max():+.2f} A")
print(f"  Runtime Change:        {rt_upg - rt_cur:+.1f} h")


# ══════════════════════════════════════════════════════════════
# 8. PLOTS
# ══════════════════════════════════════════════════════════════

# ── Plot 1: Knee Angle  [Results & Validation] ───────────────
# Purpose: proves input data is physiologically correct (30–67° range)
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(gait_percent, np.rad2deg(theta), color='#7B2D8B', linewidth=2.5)
add_window(ax)
ax.set_ylim(0, 80)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
style_ax(ax, 'Gait Cycle (%)', 'Knee Angle (degrees)',
         'Knee Angle Profile — CMU Normative Baseline')
ax.annotate(f'Peak flexion: {np.rad2deg(theta).max():.1f}°',
            xy=(gait_percent[np.argmax(theta)], np.rad2deg(theta).max()),
            xytext=(58, 72), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='gray'))
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig("plot1_knee_angle.png")
print("\nSaved plot1_knee_angle.png")

# ── Plot 2: Human Torque  [Results & Validation] ─────────────
# Purpose: shows what the knee demands — justifies assist window selection
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(gait_percent, tau_human, color=CBK, linewidth=2.5, label='Human Knee Torque')
ax.axhline(TORQUE_THRESH, color='gray', linestyle='--', linewidth=1.5,
           label=f'Assist activation threshold ({TORQUE_THRESH} Nm)')
add_window(ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
style_ax(ax, 'Gait Cycle (%)', 'Torque (Nm)',
         'Human Knee Torque Demand Over Gait Cycle')
ax.text(32, tau_human.max() * 0.88,
        f'Peak demand:\n{tau_human.max():.1f} Nm at push-off',
        fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig("plot2_human_torque.png")
print("Saved plot2_human_torque.png")

# ── Plot 3: Motor Output Torque  [Results & Validation] ──────
# Purpose: shows the device responding correctly within hardware limits
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(gait_percent, tau_human, color=CBK, linewidth=2.0,
        linestyle='--', alpha=0.5, label='Human demand (reference)')
ax.plot(gait_percent, tau_motor_cur_disp, color=CC, linewidth=2.5,
        label=f'Motor output — peak {tau_motor_cur.max():.2f} Nm')
ax.axhline(TAU_JOINT_CUR, color=CC, linestyle=':', linewidth=1.8, alpha=0.8,
           label=f'Hardware ceiling: {TAU_JOINT_CUR:.2f} Nm')
add_window(ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
style_ax(ax, 'Gait Cycle (%)', 'Torque (Nm)',
         'Motor Output Torque at Knee Joint — Current Hardware')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig("plot3_motor_torque.png")
print("Saved plot3_motor_torque.png")

# ── Plot 4: Remaining Human Effort  [Results & Validation] ───
# Purpose: primary outcome — how much work the exo offloads
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(gait_percent, tau_human,           color=CBK, linewidth=2.5,
        label='No Assist (baseline)')
ax.plot(gait_percent, tau_assist_cur_disp, color=CC,  linewidth=2.5,
        label=f'With Exoskeleton — {work_red_cur:.1f}% work reduction')
ax.fill_between(gait_percent, tau_assist_cur_disp, tau_human,
                where=mask, alpha=0.12, color=CC,
                label='Offloaded work region')
add_window(ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
style_ax(ax, 'Gait Cycle (%)', 'Torque (Nm)',
         'Remaining Human Knee Effort With Exoskeleton Assistance')
ax.text(63, tau_human.max() * 0.88,
        f'Work reduction:\n{work_red_cur:.1f}% per cycle',
        fontsize=10, fontweight='bold', color=CC,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig("plot4_remaining_effort.png")
print("Saved plot4_remaining_effort.png")

# ── Plot 5: Motor Current Draw  [Results & Validation] ───────
# Purpose: hardware validation — proves motor stays within safe limits
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(gait_percent, I_draw_cur_disp, color=CC, linewidth=2.5,
        label=f'Motor current — peak {I_draw_cur.max():.2f} A')
ax.axhline(I_cont, color='#C0392B', linestyle='--', linewidth=2.0,
           label=f'Continuous limit: {I_cont} A')
ax.fill_between(gait_percent, I_draw_cur_disp, alpha=0.10, color=CC)
add_window(ax)
ax.set_ylim(bottom=0)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
style_ax(ax, 'Gait Cycle (%)', 'Current (A)',
         'Motor Current Draw — Hardware Safety Validation')
ax.text(32, I_cont * 1.05, f'Safe limit: {I_cont} A',
        color='#C0392B', fontsize=9, fontweight='bold')
ax.text(62, I_draw_cur.max() * 0.6,
        f'Peak: {I_draw_cur.max():.2f} A\n= {100*I_draw_cur.max()/I_cont:.0f}% of limit',
        fontsize=9, color=CC,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig("plot5_current_draw.png")
print("Saved plot5_current_draw.png")

# ── Plot 6: Upgrade Comparison  [Engineering Methodology] ─────
# Purpose: shows the engineered upgrade path — framed as future work
# Two subplots side by side: remaining effort | percent assistance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Proposed Upgrade Analysis — 30:1 Cycloidal Gearbox + 6S LiPo Battery',
             fontsize=14, fontweight='bold', y=1.01)

# Left: remaining human effort both configs
ax1.plot(gait_percent, tau_human,           color=CBK, linewidth=2.2,
         label='No Assist (baseline)')
ax1.plot(gait_percent, tau_assist_cur_disp, color=CC,  linewidth=2.2,
         label=f'Current — {work_red_cur:.1f}% reduction')
ax1.plot(gait_percent, tau_assist_upg_disp, color=CU,  linewidth=2.2,
         label=f'Upgraded (30:1) — {work_red_upg:.1f}% reduction')
ax1.fill_between(gait_percent, tau_assist_upg_disp, tau_human,
                 where=mask, alpha=0.10, color=CU)
ax1.axvspan(30, 60, alpha=0.07, color=CGR)
ax1.set_xlim(0, 100)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax1.set_xlabel('Gait Cycle (%)')
ax1.set_ylabel('Torque (Nm)')
ax1.set_title('Remaining Human Effort', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True)
ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())

# Right: % assistance inside window only
ax2.plot(gait_percent[mask], pct_assist_cur[mask], color=CC, linewidth=2.2,
         label=f'Current — avg {np.mean(pct_assist_cur[mask]):.1f}%')
ax2.plot(gait_percent[mask], pct_assist_upg[mask], color=CU, linewidth=2.2,
         label=f'Upgraded — avg {np.mean(pct_assist_upg[mask]):.1f}%')
ax2.set_xlim(30, 60)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax2.set_xlabel('Gait Cycle (%)')
ax2.set_ylabel('Assistance (%)')
ax2.set_title('Percent Assistance (Assist Window)', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True)
ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())

plt.tight_layout()
plt.savefig("plot6_upgrade_comparison.png")
print("Saved plot6_upgrade_comparison.png")


# ── Plot 7: Knee Angle + Remaining Effort  [Results & Validation] ─
# Purpose: shows HOW the joint is moving at the same time as HOW HARD it's working
# Two y-axes (dual axis): angle on left, torque on right
fig, ax_angle = plt.subplots(figsize=(11, 6))
ax_torque = ax_angle.twinx()   # second y-axis sharing the same x-axis

# Knee angle — left axis, purple
ln1 = ax_angle.plot(gait_percent, np.rad2deg(theta),
                    color='#7B2D8B', linewidth=2.5, linestyle='-',
                    label='Knee Angle')
ax_angle.set_ylabel('Knee Angle (degrees)', color='#7B2D8B', fontsize=12, fontweight='bold')
ax_angle.tick_params(axis='y', labelcolor='#7B2D8B')
ax_angle.set_ylim(0, 100)
ax_angle.yaxis.set_major_locator(ticker.MultipleLocator(10))

# Torque lines — right axis
ln2 = ax_torque.plot(gait_percent, tau_human,
                     color=CBK, linewidth=2.2, linestyle='--',
                     label='No Assist (baseline)')
ln3 = ax_torque.plot(gait_percent, tau_assist_cur_disp,
                     color=CC, linewidth=2.5,
                     label=f'With Exoskeleton — {work_red_cur:.1f}% work reduction')
ax_torque.fill_between(gait_percent, tau_assist_cur_disp, tau_human,
                       where=mask, alpha=0.10, color=CC)
ax_torque.set_ylabel('Knee Torque (Nm)', color=CBK, fontsize=12, fontweight='bold')
ax_torque.tick_params(axis='y', labelcolor=CBK)
ax_torque.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax_torque.axhline(0, color='lightgray', linewidth=0.8, linestyle='-')

# Assist window shading
ax_angle.axvspan(30, 60, alpha=0.08, color=CGR)
ln4 = ax_angle.axvspan(30, 60, alpha=0, color=CGR,
                        label='Assist Window (30–60%)')  # invisible — just for legend

# Annotations
ax_angle.annotate(f'Peak flexion\n{np.rad2deg(theta).max():.1f}°',
                  xy=(gait_percent[np.argmax(theta)], np.rad2deg(theta).max()),
                  xytext=(58, 82), fontsize=8.5, color='#7B2D8B',
                  arrowprops=dict(arrowstyle='->', color='#7B2D8B', lw=1.2))
ax_torque.text(63, tau_human.max() * 0.87,
               f'Work reduction:\n{work_red_cur:.1f}% per cycle',
               fontsize=9, fontweight='bold', color=CC,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# Combined legend from both axes
all_lines = ln1 + ln2 + ln3
all_labels = [l.get_label() for l in all_lines]
# Add assist window manually
from matplotlib.patches import Patch
all_lines_handles = all_lines + [Patch(facecolor=CGR, alpha=0.3, label='Assist Window (30–60%)')]
all_labels += ['Assist Window (30–60%)']
ax_angle.legend(all_lines_handles, all_labels, loc='upper left', fontsize=9)

ax_angle.set_xlabel('Gait Cycle (%)', fontsize=12, fontweight='bold')
ax_angle.set_xlim(0, 100)
ax_angle.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax_angle.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax_angle.grid(True, alpha=0.35, linestyle='--')
ax_angle.set_title('Knee Angle & Remaining Human Effort — Exoskeleton Assistance',
                   fontsize=14, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig("plot7_angle_and_effort.png")
print("Saved plot7_angle_and_effort.png")

# ── Save CSV ─────────────────────────────────────────────────
pd.DataFrame({
    "gait_percent":    gait_percent,
    "knee_angle_deg":  np.rad2deg(theta),
    "omega_rad_s":     omega,
    "tau_human":       tau_human,
    "tau_des_cur":     tau_des_cur,
    "tau_des_upg":     tau_des_upg,
    "tau_motor_cur":   tau_motor_cur,
    "tau_motor_upg":   tau_motor_upg,
    "tau_assist_cur":  tau_assist_cur,
    "tau_assist_upg":  tau_assist_upg,
    "pct_assist_cur":  pct_assist_cur,
    "pct_assist_upg":  pct_assist_upg,
    "power_cur_W":     power_cur,
    "power_upg_W":     power_upg,
    "current_cur_A":   I_draw_cur,
    "current_upg_A":   I_draw_upg,
}).to_csv("knee_exo_results.csv", index=False)

plt.show()
print("\nDone. Saved 7 plots + knee_exo_results.csv")
