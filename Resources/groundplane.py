import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.5)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Road surface ──────────────────────────────────────────────────────────
ax.plot([0, 10], [1.0, 1.0], color='black', lw=2.2, solid_capstyle='round')
# Road centre dashes
for x in np.arange(4.5, 10, 1.4):
    ax.plot([x, x+0.7], [1.0, 1.0], color='#AAAAAA', lw=1.0, linestyle='-')

# ── Vehicle body ──────────────────────────────────────────────────────────
vx, vy, vw, vh = 0.7, 1.0, 1.9, 1.25
ax.add_patch(plt.Rectangle((vx, vy), vw, vh,
    fill=True, facecolor='#EBEBEB', edgecolor='black', lw=1.1, zorder=3))
# Windscreen
ax.add_patch(plt.Polygon(
    [(vx+vw*0.55, vy+vh), (vx+vw*0.95, vy+vh), (vx+vw*0.98, vy+vh*0.45), (vx+vw*0.55, vy+vh*0.38)],
    fill=True, facecolor='#C8C8C8', edgecolor='black', lw=0.6, zorder=4))
# Bonnet line
ax.plot([vx+vw*0.55, vx+vw], [vy+vh*0.38, vy+vh*0.38],
        color='black', lw=0.6, zorder=4)
# Wheels
for wx in [vx+vw*0.18, vx+vw*0.80]:
    ax.add_patch(plt.Circle((wx, vy), 0.19,
        fill=True, facecolor='#363636', edgecolor='black', lw=0.8, zorder=5))
    ax.add_patch(plt.Circle((wx, vy), 0.09,
        fill=True, facecolor='#A0A0A0', edgecolor='black', lw=0.5, zorder=6))

# ── Camera (small filled square at front-top of vehicle) ──────────────────
cam_x = vx + vw + 0.04
cam_y = vy + vh * 0.88
ax.add_patch(plt.Rectangle((cam_x-0.09, cam_y-0.09), 0.18, 0.18,
    fill=True, facecolor='black', edgecolor='black', lw=0.5, zorder=7))

# ── Vertical dashed height line ────────────────────────────────────────────
ax.plot([cam_x, cam_x], [vy, cam_y], color='black', lw=0.9,
        linestyle=(0, (4, 3)), zorder=2)

# ── Horizontal dotted reference line (horizon) ────────────────────────────
ax.plot([cam_x, 6.8], [cam_y, cam_y], color='black', lw=0.7,
        linestyle=(0, (2, 3)), alpha=0.75, zorder=2)

# ── Detection point on road ────────────────────────────────────────────────
det_x, det_y = 7.8, 1.0
ax.plot([det_x, det_x], [0.82, 1.18], color='black', lw=2.0, zorder=4)

# ── Main viewing ray ───────────────────────────────────────────────────────
ax.annotate('', xy=(det_x, det_y), xytext=(cam_x, cam_y),
    arrowprops=dict(arrowstyle='->', color='black', lw=1.4, mutation_scale=13), zorder=5)

# ── Angle arc at camera ────────────────────────────────────────────────────
ray_deg = np.degrees(np.arctan2(det_y - cam_y, det_x - cam_x))
arc = mpatches.Arc((cam_x, cam_y), 1.7, 1.7,
                    angle=0, theta1=ray_deg, theta2=0,
                    color='black', lw=0.9, zorder=6)
ax.add_patch(arc)

# ── Height double-arrow (left of camera) ──────────────────────────────────
bx = cam_x - 0.45
ax.annotate('', xy=(bx, vy), xytext=(bx, cam_y),
    arrowprops=dict(arrowstyle='<->', color='black', lw=0.9, mutation_scale=10), zorder=3)
ax.plot([bx-0.07, bx+0.07], [vy, vy], color='black', lw=0.8)
ax.plot([bx-0.07, bx+0.07], [cam_y, cam_y], color='black', lw=0.8)

# ── Distance double-arrow (along road, below) ─────────────────────────────
dy = 0.52
ax.annotate('', xy=(det_x, dy), xytext=(cam_x, dy),
    arrowprops=dict(arrowstyle='<->', color='black', lw=0.9, mutation_scale=10), zorder=3)
ax.plot([cam_x-0.0, cam_x], [dy-0.06, dy+0.06], color='black', lw=0.8)
ax.plot([det_x-0.0, det_x], [dy-0.06, dy+0.06], color='black', lw=0.8)

plt.tight_layout(pad=0.15)
plt.savefig('./ground_plane_geometry.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Done.")