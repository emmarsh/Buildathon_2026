import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext
import joblib
import os

# -----------------------------
# 1. Load your ML Model & Vectorizer
# -----------------------------
MODEL_FILE = 'language_model_rf.pkl'
VECTORIZER_FILE = 'vectorizer_rf.pkl'

# Fallback/Dummy logic if files aren't found yet
if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    clf = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    MODEL_ACTIVE = True
else:
    print("Warning: Model files not found. Simulation will use keyword fallback.")
    MODEL_ACTIVE = False

# Mapping model language output to our Simulation Region Names
# Adjusted to match your Training script's region_map
REGION_MAP = {
    'TAMIL': 'Tamil Nadu',
    'HINDI': 'Delhi',
    'TELUGU': 'Telangana', # Adjusted to match simulation regions
    'MALAYALAM': 'Kerala',
    'PUNJAB': 'Punjab',
    'ASSAM': 'Assam'
}

# -----------------------------
# 2. Simulation Setup
# -----------------------------
REGION_MAPPING = {
    1: "Tamil Nadu", 2: "Kerala", 3: "Telangana",
    4: "Delhi", 5: "Punjab", 6: "Assam"
}

# Key: Region Name, Value: List of received messages
broadcast_schedule = {name: [] for name in REGION_MAPPING.values()}

REGIONS_GEOM = {
    1: [(5, 5), (30, 8), (28, 25), (6, 22)],
    2: [(30, 8), (55, 6), (60, 22), (28, 25)],
    3: [(55, 6), (95, 10), (90, 25), (60, 22)],
    4: [(6, 22), (28, 25), (30, 52), (5, 48)],
    5: [(28, 25), (60, 22), (58, 52), (30, 52)],
    6: [(60, 22), (90, 25), (95, 50), (58, 52)]
}

def point_in_polygon(x, y, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        if intersect: inside = not inside
        j = i
    return inside

# Create UEs
np.random.seed(42)
UEs = []
all_coords = [p for poly in REGIONS_GEOM.values() for p in poly]
min_x, max_x = min(p[0] for p in all_coords), max(p[0] for p in all_coords)
min_y, max_y = min(p[1] for p in all_coords), max(p[1] for p in all_coords)

for i in range(100):
    l_rid = np.random.choice(list(REGION_MAPPING.keys()))
    x, y = np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)
    p_rid = next((rid for rid, poly in REGIONS_GEOM.items() if point_in_polygon(x, y, poly)), None)
    if p_rid:
        UEs.append({"UE_ID": i, "Logical_Region": REGION_MAPPING[l_rid], "Physical_Region": REGION_MAPPING[p_rid], "Position": (x, y)})

# -----------------------------
# 3. Core Logic (ML + Targeting)
# -----------------------------

def update_schedule(event=None):
    root = tk.Tk(); root.withdraw()
    text = simpledialog.askstring("Broadcast Core", "Enter content to broadcast:")
    if not text: return

    # ML INFERENCE
    if MODEL_ACTIVE:
        vec_text = vectorizer.transform([text])
        prediction = clf.predict(vec_text)[0].upper()
        target_region = REGION_MAP.get(prediction, "All")
    else:
        # Fallback keyword logic for testing
        target_region = "Delhi"
        if "வணக்கம்" in text: target_region = "Tamil Nadu"

    # Add to specific region's schedule
    if target_region == "All":
        for r in broadcast_schedule: broadcast_schedule[r].append(text)
    elif target_region in broadcast_schedule:
        broadcast_schedule[target_region].append(text)
    
    messagebox.showinfo("Broadcast Core", f"Classification: {target_region}\nStatus: Transmitted to RANs.")
    root.destroy()

def view_schedule(event=None):
    root = tk.Tk(); root.title("Live Broadcast Schedule")
    txt = scrolledtext.ScrolledText(root, width=40, height=12)
    txt.pack()
    for reg, msgs in broadcast_schedule.items():
        txt.insert(tk.END, f"[{reg}]: {len(msgs)} msgs\n")
    root.mainloop()

def on_pick(event):
    idx = event.ind[0]
    ue = UEs[idx]
    received = broadcast_schedule.get(ue['Logical_Region'], [])
    content = "\n".join([f"- {m}" for m in received]) if received else "No messages for your region."
    
    root = tk.Tk(); root.withdraw()
    messagebox.showinfo(f"UE {ue['UE_ID']} Terminal", 
                        f"Logical Home: {ue['Logical_Region']}\nPhysical Location: {ue['Physical_Region']}\n\nINBOX:\n{content}")
    root.destroy()

# -----------------------------
# 4. Visualization
# -----------------------------
fig, ax = plt.subplots(figsize=(15, 8))
plt.subplots_adjust(left=0.2) # Make room for Broadcast Core box

# Draw Broadcast Core (UI Box on the left)
core_rect = Rectangle((-50, 5), 40, 50, linewidth=2, edgecolor='black', facecolor='#f0f0f0', alpha=0.8)
ax.add_patch(core_rect)
ax.text(-30, 52, "BROADCAST CORE", fontweight='bold', ha='center', fontsize=12)

# Region Plotting
colors = plt.cm.Pastel1(np.linspace(0, 1, 6))
for i, (rid, poly) in enumerate(REGIONS_GEOM.items()):
    ax.add_patch(Polygon(poly, closed=True, color=colors[i], alpha=0.5, label=REGION_MAPPING[rid]))
    cx, cy = np.mean([p[0] for p in poly]), np.mean([p[1] for p in poly])
    ax.text(cx, cy, REGION_MAPPING[rid], fontweight='bold', ha='center', fontsize=10)
    ax.scatter(cx, cy, marker='^', s=200, color='black', zorder=5)

# Plot UEs
ue_scat = ax.scatter([u["Position"][0] for u in UEs], [u["Position"][1] for u in UEs], 
                    s=35, color='blue', alpha=0.6, picker=True, zorder=10)

# Dashboard Buttons (Inside the Broadcast Core box)
ax_upd = plt.axes([0.05, 0.6, 0.1, 0.08])
ax_vie = plt.axes([0.05, 0.45, 0.1, 0.08])
btn_upd = Button(ax_upd, 'Update\nSchedule', color='#d1e7dd')
btn_vie = Button(ax_vie, 'View\nSchedule', color='#fff3cd')

btn_upd.on_clicked(update_schedule)
btn_vie.on_clicked(view_schedule)

fig.canvas.mpl_connect('pick_event', on_pick)
ax.set_xlim(-55, 100)
ax.axis('off')
plt.show()
