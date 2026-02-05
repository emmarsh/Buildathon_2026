import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext
import joblib
import os
import pandas as pd
import datetime
import time
import requests
import json
import re
import threading
from collections import deque
from queue import Queue
import csv
BROADCAST_HISTORY = deque(maxlen=5)
AGENT_INPUT_QUEUE = Queue()
AGENT_QUEUE = deque()
DISPATCH_INTERVAL = 5  # seconds
LAST_DISPATCH_TIME = 0

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

def ollama_analyze(text):
    prompt = f"""
You are a strict JSON generator.

ONLY return valid JSON. No markdown. No comments. No explanations. Refer {ALERT_KEYWORDS} for priority hints. 
Message:
{text}

JSON schema:
{{
 "language": "Tamil|Kannada|Hindi|Punjabi|Telugu|Malayalam|Bengali|Assamese|Gujarati|English|Unknown",
 "region": "Tamil Nadu|Karnataka|Delhi|Telangana|Kerala|West Bengal|Assam|Gujarat|Andhra Pradesh|Punjab|All",
 "priority": "P1|P2|P3|P4",
 "is_alert": true|false
}}
"""

    try:
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=100)

        raw = res.json().get("response", "").strip()
        if not raw:
            return None

        try:
            return json.loads(raw)
        except:
            return None

        # HARD CLEANUP — strip code blocks if Ollama adds them
        #raw = re.sub(r"```json|```", "", raw).strip()

    except Exception as e:
        print("OLLAMA FAIL:", e)
        return None
ALERT_KEYWORDS = {
    "P1": [
        # English
        "earthquake", "tsunami", "gas leak", "evacuate", "shooter", "explosion", "fire", "Political issue",
        # Kannada
        "ಭೂಕಂಪ", "ಸ್ಫೋಟ", "ಅಗ್ನಿ", "ಬಾಂಬ್",
        # Tamil
        "நிலநடுக்கம்", "சுனாமி", "வெடிப்பு", "தீ", "வாயு கசிவு","அவசரம்",
        # Telugu
        "భూకంపం", "సునామీ", "విస్ఫోటనం", "అగ్ని",
        # Malayalam
        "ഭൂകമ്പം", "സുനാമി", "വിസ്ഫോടനം", "തീ",
        # Hindi
        "भूकंप", "सुनामी", "विस्फोट", "आग",
        # Punjabi
        "ਭੂਚਾਲ", "ਸੁਨਾਮੀ", "ਵਿਸਫੋਟ", "ਅੱਗ",
        # Bengali
        "ভূমিকম্প", "সুনামি", "বিস্ফোরণ", "আগুন","ভাইরাস"
        # Assamese
        "ভূমিকম্প", "সুনামি", "বিস্ফোৰণ", "আগুন",
        # Gujarati
        "ભૂકંપ", "સુનામી", "વિસ્ફોટ", "આગ",
    ],
    "P2": [
        "cyclone", "flood", "storm", "industrial","education","state-specific",
        "ಚಂಡಮಾರುತ", "ನೆರೆ", "ಮಳೆ",
        "புயல்", "வெள்ளம்","சூறாவளி",
        "చక్రవాతం", "వరద",
        "ചുഴലിക്കാറ്റ്", "വെള്ളപ്പൊക്കം",
        "तूफान", "बाढ़","हवा",
        "ਤੂਫ਼ਾਨ", "ਬਾਰਿਸ਼",
        "ঘূর্ণিঝড়", "বন্যা",
        "ঘূর্ণিঝড়", "বন্যা",
        "ચક્રવાત", "બારિશ",

    ],
    "P3": [
        "traffic", "road closed", "water supply", "power outage","Radio broadcast","FM","city specific",
        "ಟ್ರಾಫಿಕ್", "ನೀರಿನ ಕೊರತೆ",
        "போக்குவரத்து", "மின்தடை",
        "ట్రాఫిక్", "నీటి కొరత",
        "ഗതാഗതം", "വൈദ്യുതി തകരാർ",
        "यातायात", "बिजली कटौती",
        "ਟ੍ਰੈਫਿਕ", "ਬਿਜਲੀ ਕੱਟੌਤੀ",
        "ট্রাফিক", "বিদ্যুৎ বিভ্রাট",
        "ট্রাফিক", "বিদ্যুৎ বিভ্রাট",
        "ટ્રાફિક", "પાણી પુરવઠો",

    ],
    "P4":[
        "weather update", "general info", "community event", "festival","cultural event",
        "ಹವಾಮಾನ", "ಸಾಮಾನ್ಯ ಮಾಹಿತಿ",
        "வானிலை", "பொது தகவல்",
        "వాతావరణం", "సాధారణ సమాచారం",
        "വാനില", "പൊതുവിവരം",
        "मौसम", "सामान्य जानकारी",
        "ਮੌਸਮ", "ਆਮ ਜਾਣਕਾਰੀ",
        "আবহাওয়া", "সাধারণ তথ্য",
        "আবহাওয়া", "সাধারণ তথ্য",
        "હવામાન", "સામાન્ય માહિતી",
    ]
}
LANGUAGE_HINTS = {
    "Tamil Nadu": ["வணக்கம்", "நிலநடுக்கம்", "புயல்","Tamil"],
    "Kerala": ["വെള്ളപ്പൊക്കം", "ചുഴലിക്കാറ്റ്","Kerala","Malayalam"],
    "Karnataka": ["ಭೂಕಂಪ", "ಚಂಡಮಾರುತ","Karnataka"],
    #"Andhra Pradesh": ["భూకంపం", "చక్రవాతం"],
    "Telangana": ["భూకంపం", "చక్రవాతం", "Telugu","Telangana"],
    "Delhi": ["earthquake", "alert", "emergency", "भूकंप", "तूफान", "सुनामी","Delhi","Hindi"],
    "Gujarat": ["ભૂકંપ", "ચક્રવાત","Gujarat","Gujarati"],
    "Assam": ["ভূমিকম্প", "বন্যা","Assam","Assamese"],
    "West Bengal": ["ভূমিকম্প", "ঘূর্ণিঝড়","Bengali","West Bengal"],
    "Punjab": ["ਭੂਚਾਲ", "ਤੂਫ਼ਾਨ","Punjab","Punjabi"],
}
def detect_priority(text):
    text = text.lower()
    for p, words in ALERT_KEYWORDS.items():
        for w in words:
            if w.lower() in text:
                return p
    return "P4"
def detect_region_deterministic(text):
    text_lower = text.lower()
    for region, hints in LANGUAGE_HINTS.items():
        for h in hints:
            if h.lower() in text_lower:
                return region
    return None

def agent_decide_and_schedule(raw_content):

    # 1) Deterministic pass
    priority = detect_priority(raw_content)
    region = detect_region_deterministic(raw_content)

    # 2) Ollama fallback
    if (priority == "P4" or region is None):
        ai = ollama_analyze(raw_content)
        if ai:
            priority = ai.get("priority", priority)
            region = ai.get("region", region)

    if region is None:
        region = "All"

    # 3) Store decision
    job = {
        "content": raw_content,
        "priority": priority,
        "region": region,
        "timestamp": time.time()
    }

    AGENT_QUEUE.append(job)

    return f"Queued → {priority} → {region}"

PRIORITY_ORDER = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}

def agent_dispatch():
    global LAST_DISPATCH_TIME
    if CURRENT_SYSTEM == "TRADITIONAL":
        return
    
    if not AGENT_QUEUE:
        return

    now = time.time()
    if now - LAST_DISPATCH_TIME < DISPATCH_INTERVAL:
        return

    LAST_DISPATCH_TIME = now

    sorted_jobs = sorted(
        list(AGENT_QUEUE),
        key=lambda j: (PRIORITY_ORDER[j["priority"]], j["timestamp"])
    )

    job = sorted_jobs[0]
    AGENT_QUEUE.remove(job)

    execute_broadcast(job["content"], forced_region=job["region"])
    live_monitor_log.append(
        f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
        f"AGENT DISPATCH → {job['priority']}"
    )
CURRENT_SYSTEM = "ATSC3.0+AI"  
CSV_LOG_FILE = "atsc3_comparative_datas.csv"
if not os.path.exists(CSV_LOG_FILE):
    with open(CSV_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp",
            "System_Type",
            "Target_Region",
            "Latency_Sec",
            "Spectral_Efficiency_Pct",
            "Relevance_Pct",
            "Reliability_Pct",
            "Energy_Eff_BitsPerJ"
        ])

def log_kpi_csv(system, target, lat, eff, rel, reliab, energy):
    with open(CSV_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.now(),
            system,
            target,
            lat,
            eff,
            rel,
            reliab,
            energy
        ])


# -----------------------------
# 1. PHYSICS & NETWORK CONSTANTS
# -----------------------------
SPEED_LIGHT_FIBER = 200000.0  # km/s (Backhaul: Delhi -> Region)
SPEED_LIGHT_AIR   = 300000.0  # km/s (OTA: Tower -> UE)
ATSC3_BANDWIDTH   = 3.0 * 10**6 # 3 Mbps (Robust PLP capacity)
ATSC3_FRAMING_DELAY = 0.080 # 80ms fixed framing/interleaving delay
DISTANCES_FROM_DELHI = {
    "Delhi": 15,       
    "Punjab": 300,
    "Telangana": 1500,
    "Assam": 1900,
    "West Bengal": 1300,
    "Gujarat": 800,
    "Karnataka": 2000,
    "Tamil Nadu": 2200,
    "Kerala": 2600,
    "All": 1300        
}

# -----------------------------
# 2. MODULAR BROADCAST STRATEGY
# -----------------------------
class KPIEngine:
    def __init__(self):
        self.latencies = []
        self.total_bits = 0
        self.useful_bits = 0
        self.relevant_deliveries = 0
        self.total_deliveries = 0
        self.history_log = []

    def calculate_hero_metrics(self, packet, distance_km):
        # ATSC 3.0 CONSTANTS
        CHANNEL_BW = 6.0 * 10**6
        LDM_GAIN = 1.40 
        OVERHEAD_LOSS = 0.20
        THROUGHPUT = 2.5 * 10**6
        TX_POWER = 20.0 
        # 1. Efficiency
        total_bits = packet['header']['size_bits'] + packet['payload_size_bits']
        tx_duration = total_bits / THROUGHPUT
        raw_eff = (packet['payload_size_bits'] / (CHANNEL_BW * tx_duration)) * 100
        efficiency_pct = min(raw_eff * LDM_GAIN * (1.0 - OVERHEAD_LOSS), 99.9)
        # 2. Latency
        t_backhaul = DISTANCES_FROM_DELHI.get(packet['metadata']['target_region'], 1300) / SPEED_LIGHT_FIBER
        t_ota = 50.0 / SPEED_LIGHT_AIR
        latency = t_backhaul + 0.200 + tx_duration + t_ota
        # 3. Reliability 
        path_loss = 10 * 2.8 * np.log10(distance_km if distance_km > 0.1 else 0.1)
        rx_power = 43.0 - path_loss + np.random.normal(0, 5.0)
        snr = rx_power - (-95.0)
        reliability = (1 / (1 + np.exp(-(snr - (-5.0))))) * 100.0
        # 4. Energy
        energy = packet['payload_size_bits'] / (TX_POWER * tx_duration) if tx_duration > 0 else 0
        return latency, efficiency_pct, reliability, energy
    
    def calculate_legacy_metrics(self, packet, distance_km):
        #LEGACY CONSTANTS
        CHANNEL_BW = 8.0 * 10**6      
        LDM_GAIN = 1.0                
        OVERHEAD_LOSS = 0.25          
        THROUGHPUT = 2.0 * 10**6      
        TX_POWER = 20.0
        # 1. Efficiency
        total_bits = packet['header']['size_bits'] + packet['payload_size_bits']
        tx_duration = total_bits / THROUGHPUT
        raw_eff = (packet['payload_size_bits'] / (CHANNEL_BW * tx_duration)) * 100
        efficiency_pct = min(raw_eff * LDM_GAIN * (1.0 - OVERHEAD_LOSS), 85.0)
        # 2. Latency
        t_backhaul = DISTANCES_FROM_DELHI.get("All", 1300) / SPEED_LIGHT_FIBER
        t_ota = 50.0 / SPEED_LIGHT_AIR
        latency = t_backhaul + 0.300 + tx_duration + t_ota
        # 3. Reliability 
        path_loss = 10 * 2.8 * np.log10(distance_km if distance_km > 0.1 else 0.1)
        rx_power = 43.0 - path_loss + np.random.normal(0, 5.0)
        snr = rx_power - (-95.0)
        reliability = (1 / (1 + np.exp(-(snr - 9.0)))) * 100.0
        # 4. Energy 
        energy = packet['payload_size_bits'] / (TX_POWER * tx_duration) if tx_duration > 0 else 0
        return latency, efficiency_pct, reliability, energy
    
    def record_transmission(self, packet, latency_sec, count_intended_users, count_receiving_users, distance_km, target_region):   
        mdict = {'lat':0, 'effi':0, 'reliab':0, 'eng':0}
        if CURRENT_SYSTEM == "ATSC3.0+AI":
            latency, eff, reliability, energy = self.calculate_hero_metrics(
                packet, distance_km
            )
            #relevance = is_relevant #100.0 if packet['metadata']['target_region'] != "All" else 10.0
        else:
            latency, eff, reliability, energy = self.calculate_legacy_metrics(
                packet, distance_km
            )
        final_relevance_pct=0
        if count_intended_users > 0:
            final_relevance_pct = (count_intended_users / count_receiving_users) * 100.0
            final_relevance_pct = min(final_relevance_pct, 100.0)
        mdict['lat']+=latency
        mdict['effi']+=eff
        mdict['reliab']+=reliability
        mdict['eng']+=energy
        # ---------------------------------------------------------
        lat = mdict['lat']/count_receiving_users
        eng = mdict['eng']/count_receiving_users
        reliab = mdict['reliab']/count_receiving_users
        effi = mdict['effi']/count_receiving_users
        # Energy Penalty for Traditional: Wasted energy on non-relevant users
        if CURRENT_SYSTEM != "ATSC3.0+AI":
            energy = energy * (final_relevance_pct / 100.0)
        payload_bits = packet['payload_size_bits']
        header_bits = packet['header']['size_bits']
        total_bits = payload_bits + header_bits
        self.latencies.append(latency)
        self.total_bits += total_bits
        self.useful_bits += payload_bits
        self.total_deliveries += 1
        if final_relevance_pct > 50:
            self.relevant_deliveries += 1
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        report_entry = (
            f"[{timestamp}] BROADCAST EVENT #{len(self.history_log)+1}\n"
            f" > Mode:        {CURRENT_SYSTEM}\n"
            f" > Target:      {target_region}\n"
            f" > Latency:     {mdict['lat']*1000:.2f} ms\n"
            f" > Efficiency:  {mdict['effi']:.2f} %\n"
            f" > Reliability: {mdict['reliab']:.2f} %\n"
            f" > Relevance:   {final_relevance_pct:.1f} %\n"
            f" > Energy Eff.: {mdict['eng']:.2f} bits/J\n"
            f"----------------------------------------"
        )
        self.history_log.append(report_entry)
        log_kpi_csv(
            CURRENT_SYSTEM,
            target_region,
            mdict['lat'],
            mdict['effi'],
            final_relevance_pct,
            mdict['reliab'],
            mdict['eng']
        )

    def get_full_report(self):
        avg_lat = np.mean(self.latencies) if self.latencies else 0.0
        eff = (self.useful_bits / self.total_bits * 100) if self.total_bits > 0 else 0.0
        rel = (self.relevant_deliveries / self.total_deliveries * 100) if self.total_deliveries > 0 else 0.0
        header = (f"=== CUMULATIVE SUMMARY ===\n"
                  f"Avg Latency:  {avg_lat*1000:.1f} ms\n"
                  f"Net Efficency:{eff:.1f} %\n"
                  f"Relevance:    {rel:.1f} %\n"
                  f"==========================\n\n")
        if not self.history_log:
            return header + "(No broadcasts yet)"
        stack = "\n".join(reversed(self.history_log))
        return header + stack
        
class BroadcastSystem:
    def encapsulate(self, payload, target_region, service_id):
        raise NotImplementedError

class ATSC3_Strategy(BroadcastSystem):
    def __init__(self):
        self.tsi_counter = 100 
        self.toi_counter = 1    

    def encapsulate(self, payload, target_region, service_id):
        timestamp = time.time()
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
# 4. SETUP & ML context
# -----------------------------
MODEL_FILE, VECTORIZER_FILE = 'language_model_rf.pkl', 'vectorizer_rf.pkl'
MODEL_ACTIVE = False
if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    try:
        clf = joblib.load(MODEL_FILE); vectorizer = joblib.load(VECTORIZER_FILE)
        MODEL_ACTIVE = True
    except: pass
REGION_MAP_ML = {'TAMIL':'Tamil Nadu', 'HINDI':'Delhi','GUJARATI':'Gujarat', 'TELUGU':'Telangana', 'MALAYALAM':'Kerala','KANNADA':'Karnataka','BENGALI':'West Bengal', 'PUNJAB':'Punjab', 'ASSAM':'Assam', 'ENGLISH':'All'}
REGION_MAPPING = {1: "Tamil Nadu", 2: "Kerala", 3: "Telangana", 4: "Delhi", 5: "Punjab", 6: "Assam", 7: "Gujarat", 8: "West Bengal", 9: "Karnataka"}
REGIONS_GEOM = {
    1: [(5, 5), (30, 8), (28, 25), (6, 22)],
    2: [(30, 8), (55, 6), (60, 22), (28, 25)],
    3: [(55, 6), (95, 10), (90, 25), (60, 22)],
    4: [(6, 22), (28, 25), (30, 52), (5, 48)],
    5: [(28, 25), (60, 22), (58, 52), (30, 52)],
    6: [(60, 22), (90, 25), (95, 50), (58, 52)],
    7: [(5, 48), (30, 52), (28, 65), (5, 62)],
    8: [(30, 52), (58, 52), (60, 65), (28, 65)],
    9: [(58, 52), (95, 50), (90, 65), (60, 65)]
    #10: [(55, 6), (95, 10), (90, 25), (60, 22)]
}
broadcast_system = ATSC3_Strategy() 
kpi_engine = KPIEngine()
broadcast_queue = AGENT_QUEUE
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
# 5. Adding to Agent Queue and GUI
# -----------------------------
def add_to_queue(event=None):
    root = tk.Tk(); root.withdraw()
    content = simpledialog.askstring("Scheduler", "Enter content:")
    if content:
        if CURRENT_SYSTEM == "TRADITIONAL":
            execute_broadcast(content)
            messagebox.showinfo("Broadcast", "Traditional broadcast sent immediately.")
        else:
            AGENT_INPUT_QUEUE.put(content)
            messagebox.showinfo("Agent", "Message submitted to agent.")
    root.destroy()

def agent_worker():
    while True:
        raw_content = AGENT_INPUT_QUEUE.get()
        if CURRENT_SYSTEM == "TRADITIONAL":
            AGENT_INPUT_QUEUE.task_done()
            continue
        priority = detect_priority(raw_content)
        region = detect_region_deterministic(raw_content)

        if priority == "P4" or region is None:
            ai = ollama_analyze(raw_content)
            if ai:
                priority = ai.get("priority", priority)
                region = ai.get("region", region)

        if region is None:
            for region_name, hints in LANGUAGE_HINTS.items():
                if any(word in raw_content for word in hints):
                    region = region_name
                    break
        if region is None:
            region = "All"

        job = {
            "content": raw_content,
            "priority": priority,
            "region": region,
            "timestamp": time.time()
        }

        AGENT_QUEUE.append(job)
        live_monitor_log.append(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
            f"AGENT QUEUED → {priority}"
        )

        AGENT_INPUT_QUEUE.task_done()


def execute_broadcast(raw_content,forced_region=None):
    target_region = forced_region or "All"
    if CURRENT_SYSTEM != "TRADITIONAL":
        if MODEL_ACTIVE:
            try: target_region = REGION_MAP_ML.get(clf.predict(vectorizer.transform([raw_content]))[0].upper(), "All")
            except: pass
        else:
            if any(w in raw_content for w in ["வணக்கம்", "Tamil"]): target_region = "Tamil Nadu"
            elif any(w in raw_content for w in ["Kerala", "Malayalam"]): target_region = "Kerala"
            elif any(w in raw_content for w in ["Karnataka", "Kannada"]): target_region = "Karnataka"
            elif any(w in raw_content for w in ["Bengali", "West Bengal"]): target_region = "West Bengal"
            elif any(w in raw_content for w in ["Delhi", "Hindi"]): target_region = "Delhi"
            elif any(w in raw_content for w in ["Punjab"]): target_region = "Punjab"
            elif any(w in raw_content for w in ["Assam"]): target_region = "Assam"
            elif any(w in raw_content for w in ["Telangana","Telugu"]): target_region = "Telangana"
    else:
        target_region = "All"
    #metrics_accum = {'lat':0, 'eff':0, 'reliab':0, 'eng':0}
    service_id = next((rid for rid, name in REGION_MAPPING.items() if name == target_region), 99)
    packet = broadcast_system.encapsulate(raw_content, target_region, service_id)
    backhaul_km = DISTANCES_FROM_DELHI.get(target_region, 1300) 
    t_backhaul = backhaul_km / SPEED_LIGHT_FIBER
    t_frame = ATSC3_FRAMING_DELAY
    total_bits = packet['header']['size_bits'] + packet['payload_size_bits']
    t_trans = total_bits / ATSC3_BANDWIDTH
    count_intended_users = 0 
    count_receiving_users = 0 
    tower_pos = (0,0)
    if target_region != "All":
        rid = next((k for k,v in REGION_MAPPING.items() if v == target_region), 1)
        tower_pos = get_region_centroid(rid)
    receivers = 0
    final_latency = 0 
    for ue in UEs:
        if ue["Physical_Region"] == target_region:
            count_intended_users += 1
        if target_region == "All" or ue["Physical_Region"] == target_region:
            count_receiving_users+= 1
            ota_dist = np.sqrt((ue["Position"][0] - tower_pos[0])**2 + (ue["Position"][1] - tower_pos[1])**2)
            t_ota = ota_dist / SPEED_LIGHT_AIR
            total_latency = t_backhaul + t_frame + t_trans + t_ota
            final_latency = total_latency
            ue_inboxes[ue["UE_ID"]].append(packet)
            receivers += 1
    if receivers == 0: final_latency = t_backhaul + t_frame + t_trans 
    
    kpi_engine.record_transmission(packet, final_latency, distance_km=ota_dist,count_intended_users=count_intended_users, count_receiving_users=count_receiving_users, target_region=target_region)
    live_monitor_log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Sent to {target_region} ({backhaul_km}km) | Lat: {final_latency*1000:.1f}ms")
    # Save to history (last 5)
    BROADCAST_HISTORY.append({
        "content": raw_content,
        "region": target_region,
        "time": datetime.datetime.now().strftime("%H:%M:%S")
    })
    return target_region, receivers

def open_cart(event=None):
    '''if not AGENT_QUEUE:
        root = tk.Tk(); root.withdraw()
        messagebox.showinfo("Info", "Agent queue is empty.")
        root.destroy()
        return'''
    win = tk.Tk()
    win.title("Agent Broadcast Queue")
    win.geometry("520x350")
    frame = tk.Frame(win)
    frame.pack(fill="both", expand=True)
    pending_inputs = list(AGENT_INPUT_QUEUE.queue)
    processed_jobs = list(AGENT_QUEUE)
    if pending_inputs:
        tk.Label(
            frame,
            text="Pending (Awaiting Agent Analysis)",
            fg="orange",
            font=("Bold", 10)
        ).pack(anchor="w", pady=(5, 2))
        for i, msg in enumerate(pending_inputs):
            tk.Label(
                frame,
                text=f"- {msg[:40]}...",
                fg="gray"
            ).pack(anchor="w")
    def refresh():
        for w in frame.winfo_children():
            w.destroy()
        if processed_jobs:
            tk.Label(
                frame,
                text="Agent Scheduled Broadcasts",
                fg="green",
                font=("Bold", 10)
            ).pack(anchor="w", pady=(10, 2))

        sorted_jobs = sorted(
            list(AGENT_QUEUE),
            key=lambda j: (PRIORITY_ORDER[j["priority"]], j["timestamp"])
        )

        for i, job in enumerate(sorted_jobs):
            age = int(time.time() - job["timestamp"])
            row = tk.Frame(frame)
            row.pack(fill="x", pady=2)

            label = f"{i+1}. [{job['priority']}] {job['region']} | {age}s | {job['content'][:25]}..."
            tk.Label(row, text=label).pack(side="left")

            tk.Button(row, text="Force Send",
                      command=lambda j=job: force_send(j),
                      bg="#c8e6c9").pack(side="right")

            tk.Button(row, text="Drop",
                      command=lambda j=job: drop_job(j),
                      bg="#ffcdd2").pack(side="right")
        if BROADCAST_HISTORY:
            tk.Label(
                    frame,
                    text="Recent Broadcasts (Last 5)",
                    fg="blue",
                    font=("Bold", 10)
            ).pack(anchor="w", pady=(10, 2))
            for i, msg in enumerate(reversed(BROADCAST_HISTORY)):
                label = f"{i+1}. [{msg['time']}] {msg['region']} | {msg['content'][:40]}..."
                tk.Label(frame, text=label, fg="gray").pack(anchor="w")
    def auto_refresh():
        refresh()
        win.after(1000, auto_refresh)
    auto_refresh()

    def force_send(job):
        AGENT_QUEUE.remove(job)
        execute_broadcast(job["content"])
        refresh()

    def drop_job(job):
        AGENT_QUEUE.remove(job)
        refresh()

    refresh()
    win.mainloop()

def toggle_mode(event=None):
    global CURRENT_SYSTEM
    CURRENT_SYSTEM = "TRADITIONAL" if CURRENT_SYSTEM == "ATSC3.0+AI" else "ATSC3.0+AI"
    messagebox.showinfo("Mode Changed", f"Current Mode: {CURRENT_SYSTEM}")

def open_monitor(event=None):
    root = tk.Tk(); root.title("Broadcast Monitor & KPIs")
    frame_left = tk.Frame(root)
    frame_left.pack(side="left", padx=10, pady=10, fill="y")
    tk.Label(frame_left, text="Live Protocol Analyzer", font=("Bold")).pack()
    txt = scrolledtext.ScrolledText(frame_left, width=45, height=25, bg="black", fg="#00ff00", font=("Consolas", 9))
    txt.pack()
    for log in live_monitor_log: txt.insert(tk.END, log + "\n")
    txt.see(tk.END)
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


def plot_comparison(event=None):
    df = pd.read_csv(CSV_LOG_FILE)
    if df['System_Type'].nunique() < 2:
        messagebox.showwarning("Insufficient Data", "Run broadcasts in BOTH modes first.")
        return
    summary = df.groupby("System_Type").mean(numeric_only=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("ATSC 3.0 vs Traditional Broadcast Comparison")
    axes[0,0].bar(summary.index, summary['Spectral_Efficiency_Pct'])
    axes[0,0].set_title("Spectral Efficiency (%)")
    axes[0,1].bar(summary.index, summary['Relevance_Pct'])
    axes[0,1].set_title("Content Relevance (%)")
    axes[0,2].bar(summary.index, summary['Latency_Sec'] * 1000)
    axes[0,2].set_title("Latency (ms)")
    axes[1,0].bar(summary.index, summary['Reliability_Pct'])
    axes[1,0].set_title("Reliability (%)")
    axes[1,1].bar(summary.index, summary['Energy_Eff_BitsPerJ'])
    axes[1,1].set_title("Energy Efficiency (bits/J)")
    axes[1,2].axis("off")
    plt.tight_layout()
    plt.show()

# -----------------------------
# 6. VISUALIZATION
# -----------------------------
fig = plt.figure(figsize=(14, 8))
core_sidebar = Rectangle((0.02, 0.1), 0.20, 0.8, transform=fig.transFigure, facecolor='#e0f7fa', edgecolor='#006064', linewidth=2, zorder=0)
fig.patches.append(core_sidebar)
fig.text(0.12, 0.86, "BROADCAST CORE", ha='center', fontsize=12, fontweight='bold', color='#006064')
ax = fig.add_axes([0.28, 0.1, 0.70, 0.8]) 
ax_add = plt.axes([0.045, 0.70, 0.15, 0.06])
ax_cart = plt.axes([0.045, 0.60, 0.15, 0.06])
ax_mon = plt.axes([0.045, 0.50, 0.15, 0.06])

btn_add = Button(ax_add, 'Add Content (+)', color='#b2dfdb', hovercolor='#80cbc4')
btn_cart = Button(ax_cart, 'View Schedule', color='#ffe0b2', hovercolor='#ffcc80')
btn_mon = Button(ax_mon, 'Monitor & KPIs', color='#fff9c4', hovercolor='#fff59d')
ax_plot = plt.axes([0.045, 0.40, 0.15, 0.06])
btn_plot = Button(ax_plot, 'Compare Graphs', color='#e1bee7')
ax_mode = plt.axes([0.045, 0.30, 0.15, 0.06])
btn_mode = Button(ax_mode, 'Toggle Mode', color='#c8e6c9')
btn_add.on_clicked(add_to_queue)
btn_cart.on_clicked(open_cart)
btn_mon.on_clicked(open_monitor)
btn_plot.on_clicked(plot_comparison)
btn_mode.on_clicked(toggle_mode)
colors = plt.cm.Set3(np.linspace(0, 1, 10))
for i, (rid, poly) in enumerate(REGIONS_GEOM.items()):
    ax.add_patch(Polygon(poly, closed=True, color=colors[i], alpha=0.5))
    cx, cy = np.mean([p[0] for p in poly]), np.mean([p[1] for p in poly])
    ax.text(cx, cy, REGION_MAPPING[rid], fontsize=8, ha='center', fontweight='bold')
    ax.scatter(cx, cy+4, marker='^', s=100, c='black', zorder=2)
ax.scatter([u["Position"][0] for u in UEs], [u["Position"][1] for u in UEs], s=30, c='blue', picker=True, zorder=10, label="UEs")
fig.canvas.mpl_connect('pick_event', on_pick)
threading.Thread(target=agent_worker, daemon=True).start()
def agent_loop():
    agent_dispatch()
timer = fig.canvas.new_timer(interval=1000)
timer.add_callback(agent_loop)
timer.start()
ax.set_xlim(-5, 105)
ax.set_ylim(-5, 65)
ax.set_title("Broadcast Simulation", fontsize=14)
ax.axis('off')
plt.show()


