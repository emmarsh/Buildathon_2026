import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext
import joblib
import os
import datetime
import time

# -----------------------------
# 1. PHYSICS & NETWORK CONSTANTS
# -----------------------------
SPEED_LIGHT_FIBER = 200000.0  # km/s (Backhaul: Delhi -> Region)
SPEED_LIGHT_AIR   = 300000.0  # km/s (OTA: Tower -> UE)
ATSC3_BANDWIDTH   = 3.0 * 10**6 # 3 Mbps (Robust PLP capacity)
ATSC3_FRAMING_DELAY = 0.080 # 80ms fixed framing/interleaving delay

# Real-world distances from Delhi (The Core) to State Centers (in km)
DISTANCES_FROM_DELHI = {
    "Delhi": 15,       
    "Punjab": 300,
    "Telangana": 1500,
    "Assam": 1900,
    "Tamil Nadu": 2200,
    "Kerala": 2600,
    "All": 1300        
}

# -----------------------------
# 2. MODULAR BROADCAST STRATEGY
# -----------------------------
class BroadcastSystem:
    def encapsulate(self, payload, target_region, service_id):
        raise NotImplementedError

class ATSC3_Strategy(BroadcastSystem):
    def __init__(self):
        self.tsi_counter = 100 
        self.toi_counter = 1    

    def encapsulate(self, payload, target_region, service_id):
        timestamp = time.time()
        # Header Overhead: IP(20)+UDP(8)+LCT(16)+ALP(4)+PHY_Sig(var) ~= 600 bits
        header_bits = 600 
        payload_bits = len(payload.encode('utf-8')) * 8
        
        packet = {
            "protocol": "ATSC3.0/ROUTE",
            "header": {
                "tsi": self.tsi_counter,
                "toi": self.toi_counter,
                "service_id": service_id,
                "timestamp": timestamp,
                "size_bits": header_bits
            },
            "metadata": {"target_region": target_region},
            "payload": payload,
            "payload_size_bits": payload_bits
        }
        self.toi_counter += 1
        return packet

# -----------------------------
# 3. KPI ANALYTICS ENGINE (STACKED VERSION)
# -----------------------------
class KPITracker:
    def __init__(self):
        # Global Aggregates
        self.latencies = []
        self.total_bits = 0
        self.useful_bits = 0
        self.relevant_deliveries = 0
        self.total_deliveries = 0
        
        # Transaction History (The Stack)
        self.history_log = [] 

    def record_transmission(self, packet, latency_sec, is_relevant, target_region):
        # Update Aggregates
        self.latencies.append(latency_sec)
        pkt_total = packet['header']['size_bits'] + packet['payload_size_bits']
        self.total_bits += pkt_total
        self.useful_bits += packet['payload_size_bits']
        self.total_deliveries += 1
        if is_relevant: self.relevant_deliveries += 1
        
        # Create Single Event Report (Transaction Receipt)
        efficiency = (packet['payload_size_bits'] / pkt_total * 100)
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        
        # Format: [Time] Target | Latency | Efficiency
        report_entry = (f"[{timestamp}] BROADCAST EVENT #{len(self.history_log)+1}\n"
                        f" > Target:    {target_region}\n"
                        f" > Latency:   {latency_sec*1000:.1f} ms\n"
                        f" > Efficency: {efficiency:.1f}%\n"
                        f" > Payload:   \"{packet['payload'][:15]}...\"\n"
                        f"----------------------------------------")
        
        # Add to stack 
        self.history_log.append(report_entry)

    def get_full_report(self):
        # 1. Global Stats
        avg_lat = np.mean(self.latencies) if self.latencies else 0.0
        eff = (self.useful_bits / self.total_bits * 100) if self.total_bits > 0 else 0.0
        rel = (self.relevant_deliveries / self.total_deliveries * 100) if self.total_deliveries > 0 else 0.0
        
        header = (f"=== CUMULATIVE SUMMARY ===\n"
                  f"Avg Latency:  {avg_lat*1000:.1f} ms\n"
                  f"Net Efficency:{eff:.1f} %\n"
                  f"Relevance:    {rel:.1f} %\n"
                  f"==========================\n\n")
        
        # 2. The Stack (Reversed so newest is top)
        if not self.history_log:
            return header + "(No broadcasts yet)"
            
        stack = "\n".join(reversed(self.history_log))
        return header + stack

# -----------------------------
# 4. SETUP & GLOBAL STATE
# -----------------------------
MODEL_FILE, VECTORIZER_FILE = 'language_model_rf.pkl', 'vectorizer_rf.pkl'
MODEL_ACTIVE = False
if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    try:
        clf = joblib.load(MODEL_FILE); vectorizer = joblib.load(VECTORIZER_FILE)
        MODEL_ACTIVE = True
    except: pass

REGION_MAP_ML = {'TAMIL':'Tamil Nadu', 'HINDI':'Delhi', 'TELUGU':'Telangana', 'MALAYALAM':'Kerala', 'PUNJAB':'Punjab', 'ASSAM':'Assam', 'ENGLISH':'All'}
REGION_MAPPING = {1: "Tamil Nadu", 2: "Kerala", 3: "Telangana", 4: "Delhi", 5: "Punjab", 6: "Assam"}
REGIONS_GEOM = {
    1: [(5, 5), (30, 8), (28, 25), (6, 22)],
    2: [(30, 8), (55, 6), (60, 22), (28, 25)],
    3: [(55, 6), (95, 10), (90, 25), (60, 22)],
    4: [(6, 22), (28, 25), (30, 52), (5, 48)],
    5: [(28, 25), (60, 22), (58, 52), (30, 52)],
    6: [(60, 22), (90, 25), (95, 50), (58, 52)]
}

broadcast_system = ATSC3_Strategy() 
kpi_engine = KPITracker()
broadcast_queue = []
ue_inboxes = {}
live_monitor_log = []

def get_region_centroid(rid):
    poly = REGIONS_GEOM[rid]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (np.mean(xs), np.mean(ys))

def point_in_polygon(x, y, poly):
    inside = False; n = len(poly); j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        if intersect: inside = not inside
        j = i 
    return inside

np.random.seed(42)
UEs = []
all_coords = [p for poly in REGIONS_GEOM.values() for p in poly]
for i in range(80):
    x, y = np.random.uniform(min(p[0] for p in all_coords), max(p[0] for p in all_coords)), np.random.uniform(min(p[1] for p in all_coords), max(p[1] for p in all_coords))
    p_rid = next((rid for rid, poly in REGIONS_GEOM.items() if point_in_polygon(x, y, poly)), None)
    if p_rid:
        UEs.append({"UE_ID": i, "Physical_Region": REGION_MAPPING[p_rid], "Position": (x, y)})
        ue_inboxes[i] = []

print(f"DEBUG: Generated {len(UEs)} UEs.") 

# -----------------------------
# 5. CONTROLLER LOGIC (WITH PHYSICS & STACKING)
# -----------------------------
def add_to_queue(event=None):
    root = tk.Tk(); root.withdraw()
    content = simpledialog.askstring("Scheduler", "Enter content:")
    if content:
        broadcast_queue.append(content)
        messagebox.showinfo("Success", "Added to Broadcast Schedule!")
    root.destroy()

def execute_broadcast(raw_content):
    # 1. Classification
    target_region = "All"
    if MODEL_ACTIVE:
        try: target_region = REGION_MAP_ML.get(clf.predict(vectorizer.transform([raw_content]))[0].upper(), "All")
        except: pass
    else:
        if any(w in raw_content for w in ["வணக்கம்", "Tamil"]): target_region = "Tamil Nadu"
        elif any(w in raw_content for w in ["Kerala", "Malayalam"]): target_region = "Kerala"
        elif any(w in raw_content for w in ["Delhi", "Hindi"]): target_region = "Delhi"
        elif any(w in raw_content for w in ["Punjab"]): target_region = "Punjab"
        elif any(w in raw_content for w in ["Assam"]): target_region = "Assam"
        elif any(w in raw_content for w in ["Telangana"]): target_region = "Telangana"

    # 2. Encapsulation
    service_id = next((rid for rid, name in REGION_MAPPING.items() if name == target_region), 99)
    packet = broadcast_system.encapsulate(raw_content, target_region, service_id)
    
    # 3. PHYSICS CALCULATIONS
    backhaul_km = DISTANCES_FROM_DELHI.get(target_region, 1300) 
    t_backhaul = backhaul_km / SPEED_LIGHT_FIBER
    t_frame = ATSC3_FRAMING_DELAY
    total_bits = packet['header']['size_bits'] + packet['payload_size_bits']
    t_trans = total_bits / ATSC3_BANDWIDTH

    tower_pos = (0,0)
    if target_region != "All":
        rid = next((k for k,v in REGION_MAPPING.items() if v == target_region), 1)
        tower_pos = get_region_centroid(rid)

    receivers = 0
    final_latency = 0 # To store the last valid latency for logging

    for ue in UEs:
        if target_region == "All" or ue["Physical_Region"] == target_region:
            
            ota_dist = np.sqrt((ue["Position"][0] - tower_pos[0])**2 + (ue["Position"][1] - tower_pos[1])**2)
            t_ota = ota_dist / SPEED_LIGHT_AIR
            total_latency = t_backhaul + t_frame + t_trans + t_ota
            final_latency = total_latency

            ue_inboxes[ue["UE_ID"]].append(packet)
            receivers += 1
            
    # Record to Stack (Log once per broadcast event, not per user)
    # If no receivers, we still log it but latency might be theoretical
    if receivers == 0: final_latency = t_backhaul + t_frame + t_trans 
    
    kpi_engine.record_transmission(packet, final_latency, is_relevant=True, target_region=target_region)
            
    live_monitor_log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Sent to {target_region} ({backhaul_km}km) | Lat: {final_latency*1000:.1f}ms")
    return target_region, receivers

def open_cart(event=None):
    if not broadcast_queue:
        root = tk.Tk(); root.withdraw(); messagebox.showinfo("Info", "Schedule is empty."); root.destroy(); return
    win = tk.Tk(); win.title("Broadcast Schedule"); win.geometry("400x350")
    frame = tk.Frame(win); frame.pack(fill="both", expand=True)
    def refresh():
        for w in frame.winfo_children(): w.destroy()
        for i, c in enumerate(broadcast_queue):
            row = tk.Frame(frame); row.pack(fill="x", pady=2)
            tk.Label(row, text=f"{i+1}. {c[:20]}..").pack(side="left")
            tk.Button(row, text="Broadcast", command=lambda idx=i: send_item(idx), bg="#c8e6c9").pack(side="right")
            tk.Button(row, text="Remove", command=lambda idx=i: del_item(idx), bg="#ffcdd2").pack(side="right")
    def del_item(idx): broadcast_queue.pop(idx); refresh()
    def send_item(idx):
        c = broadcast_queue.pop(idx)
        r, cnt = execute_broadcast(c)
        refresh()
        messagebox.showinfo("Report", f"Broadcasted to {r}\nReceivers: {cnt}")
    refresh(); win.mainloop()

def open_monitor(event=None):
    root = tk.Tk(); root.title("Broadcast Monitor & KPIs")
    
    # SPLIT SCREEN UI
    # Left: Live Protocol Stream
    frame_left = tk.Frame(root)
    frame_left.pack(side="left", padx=10, pady=10, fill="y")
    tk.Label(frame_left, text="Live Protocol Analyzer", font=("Bold")).pack()
    txt = scrolledtext.ScrolledText(frame_left, width=45, height=25, bg="black", fg="#00ff00", font=("Consolas", 9))
    txt.pack()
    for log in live_monitor_log: txt.insert(tk.END, log + "\n")
    txt.see(tk.END)
    
    # Right: Stacked KPI Report
    frame_right = tk.Frame(root)
    frame_right.pack(side="right", padx=10, pady=10, fill="y")
    tk.Label(frame_right, text="KPI History Stack", font=("Bold")).pack()
    kpi_txt = scrolledtext.ScrolledText(frame_right, width=45, height=25, bg="#f0f0f0", font=("Consolas", 9))
    kpi_txt.pack()
    kpi_txt.insert(tk.END, kpi_engine.get_full_report())
    
    root.mainloop()

def on_pick(event):
    ue = UEs[event.ind[0]]
    inbox = ue_inboxes[ue['UE_ID']]
    content = "".join([f"[{p['protocol']}] {p['payload'][:20]}...\n" for p in inbox])
    root = tk.Tk(); root.withdraw(); messagebox.showinfo(f"UE {ue['UE_ID']}", f"Region: {ue['Physical_Region']}\n\n{content}"); root.destroy()

# -----------------------------
# 6. VISUALIZATION
# -----------------------------
fig = plt.figure(figsize=(14, 8))

# Draw Sidebar (Broadcast Core)
core_sidebar = Rectangle((0.02, 0.1), 0.20, 0.8, transform=fig.transFigure, 
                         facecolor='#e0f7fa', edgecolor='#006064', linewidth=2, zorder=0)
fig.patches.append(core_sidebar)
fig.text(0.12, 0.86, "BROADCAST CORE", ha='center', fontsize=12, fontweight='bold', color='#006064')
#fig.text(0.12, 0.83, "Origin: DELHI (Sim)", ha='center', fontsize=9, style='italic', color='#d32f2f')

# Add Map Axes
ax = fig.add_axes([0.28, 0.1, 0.70, 0.8]) 

# Draw Buttons (Inside Sidebar)
ax_add = plt.axes([0.045, 0.70, 0.15, 0.06])
ax_cart = plt.axes([0.045, 0.60, 0.15, 0.06])
ax_mon = plt.axes([0.045, 0.50, 0.15, 0.06])

btn_add = Button(ax_add, 'Add Content (+)', color='#b2dfdb', hovercolor='#80cbc4')
btn_cart = Button(ax_cart, 'View Schedule', color='#ffe0b2', hovercolor='#ffcc80')
btn_mon = Button(ax_mon, 'Monitor & KPIs', color='#fff9c4', hovercolor='#fff59d')

btn_add.on_clicked(add_to_queue)
btn_cart.on_clicked(open_cart)
btn_mon.on_clicked(open_monitor)

# Draw Map
colors = plt.cm.Set3(np.linspace(0, 1, 6))
for i, (rid, poly) in enumerate(REGIONS_GEOM.items()):
    ax.add_patch(Polygon(poly, closed=True, color=colors[i], alpha=0.5))
    cx, cy = np.mean([p[0] for p in poly]), np.mean([p[1] for p in poly])
    ax.text(cx, cy, REGION_MAPPING[rid], fontsize=8, ha='center', fontweight='bold')
    ax.scatter(cx, cy+4, marker='^', s=100, c='black', zorder=2)

# Draw UEs
ax.scatter([u["Position"][0] for u in UEs], [u["Position"][1] for u in UEs], 
           s=30, c='blue', picker=True, zorder=10, label="UEs")

fig.canvas.mpl_connect('pick_event', on_pick)


ax.set_xlim(-5, 105)
ax.set_ylim(-5, 65)
ax.set_title("Broadcast Simulation", fontsize=14)
ax.axis('off')
plt.show()
