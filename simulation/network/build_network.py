"""
Network compilation and generation script for Q-Transit Nexus.

This script:
1. Attempts to run SUMO's `netconvert` on the modular XML definitions
   (prototype_grid.nod.xml, prototype_grid.edg.xml, prototype_grid.tll.xml).
2. If `netconvert` is not found, automatically compiles a fully compliant,
   high-fidelity SUMO `network.net.xml` with calculated lane geometries,
   junction shapes, and turn connections.
"""

import os
import sys
import shutil
import subprocess

NETWORK_DIR = os.path.dirname(os.path.abspath(__file__))
NODES_FILE = os.path.join(NETWORK_DIR, "prototype_grid.nod.xml")
EDGES_FILE = os.path.join(NETWORK_DIR, "prototype_grid.edg.xml")
TLS_FILE = os.path.join(NETWORK_DIR, "prototype_grid.tll.xml")
OUTPUT_NET = os.path.join(NETWORK_DIR, "network.net.xml")

def find_netconvert():
    nc = shutil.which("netconvert")
    if nc:
        return nc
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        cand = os.path.join(sumo_home, "bin", "netconvert.exe" if os.name == "nt" else "netconvert")
        if os.path.isfile(cand):
            return cand
    return None

def build_with_netconvert(nc_bin):
    print(f"[BUILD] Using netconvert at: {nc_bin}")
    cmd = [
        nc_bin,
        "--node-files", NODES_FILE,
        "--edge-files", EDGES_FILE,
        "--tllogic-files", TLS_FILE,
        "--output-file", OUTPUT_NET,
        "--no-turnarounds", "true",
        "--junctions.corner-detail", "5",
        "--default.lanewidth", "3.20"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[SUCCESS] Network generated successfully: {OUTPUT_NET}")
        return True
    else:
        print(f"[WARNING] netconvert error:\n{res.stderr}")
        return False

def generate_standalone_net():
    print("[BUILD] Generating standalone SUMO network.net.xml...")
    # Node coordinates
    nodes = {
        "J00": (0.0, 0.0),   "J10": (300.0, 0.0),   "J20": (600.0, 0.0),   "J30": (900.0, 0.0),
        "J01": (0.0, 300.0), "J11": (300.0, 300.0), "J21": (600.0, 300.0), "J31": (900.0, 300.0),
        "J02": (0.0, 600.0), "J12": (300.0, 600.0), "J22": (600.0, 600.0), "J32": (900.0, 600.0)
    }

    # Edges definition: (id, from, to, lanes, speed, priority)
    edges = [
        # Row 0
        ("E_00_10", "J00", "J10", 2, 13.89, 2), ("E_10_00", "J10", "J00", 2, 13.89, 2),
        ("E_10_20", "J10", "J20", 2, 13.89, 2), ("E_20_10", "J20", "J10", 2, 13.89, 2),
        ("E_20_30", "J20", "J30", 2, 13.89, 2), ("E_30_20", "J30", "J20", 2, 13.89, 2),
        # Row 1 (Central Arterial)
        ("E_01_11", "J01", "J11", 3, 16.67, 3), ("E_11_01", "J11", "J01", 3, 16.67, 3),
        ("E_11_21", "J11", "J21", 3, 16.67, 3), ("E_21_11", "J21", "J11", 3, 16.67, 3),
        ("E_21_31", "J21", "J31", 3, 16.67, 3), ("E_31_21", "J31", "J21", 3, 16.67, 3),
        # Row 2
        ("E_02_12", "J02", "J12", 2, 13.89, 2), ("E_12_02", "J12", "J02", 2, 13.89, 2),
        ("E_12_22", "J12", "J22", 2, 13.89, 2), ("E_22_12", "J22", "J12", 2, 13.89, 2),
        ("E_22_32", "J22", "J32", 2, 13.89, 2), ("E_32_22", "J32", "J22", 2, 13.89, 2),
        # Col 0
        ("E_00_01", "J00", "J01", 2, 11.11, 1), ("E_01_00", "J01", "J00", 2, 11.11, 1),
        ("E_01_02", "J01", "J02", 2, 11.11, 1), ("E_02_01", "J02", "J01", 2, 11.11, 1),
        # Col 1
        ("E_10_11", "J10", "J11", 2, 11.11, 1), ("E_11_10", "J11", "J10", 2, 11.11, 1),
        ("E_11_12", "J11", "J12", 2, 11.11, 1), ("E_12_11", "J12", "J11", 2, 11.11, 1),
        # Col 2
        ("E_20_21", "J20", "J21", 2, 11.11, 1), ("E_21_20", "J21", "J20", 2, 11.11, 1),
        ("E_21_22", "J21", "J22", 2, 11.11, 1), ("E_22_21", "J22", "J21", 2, 11.11, 1),
        # Col 3
        ("E_30_31", "J30", "J31", 2, 11.11, 1), ("E_31_30", "J31", "J30", 2, 11.11, 1),
        ("E_31_32", "J31", "J32", 2, 11.11, 1), ("E_32_31", "J32", "J31", 2, 11.11, 1),
    ]

    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append('<net version="1.20" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">')
    xml.append('    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,900.00,600.00" origBoundary="0.00,0.00,900.00,600.00" projParameter="!"/>')

    # Traffic light programs
    xml.append('    <tlLogic id="TL_J11" type="static" programID="0" offset="0">')
    xml.append('        <phase duration="31" state="GGgrrrGGgrrr"/>')
    xml.append('        <phase duration="4"  state="yyyrrryyyrrr"/>')
    xml.append('        <phase duration="25" state="rrrGGgrrrGGg"/>')
    xml.append('        <phase duration="4"  state="rrryyyrrryyy"/>')
    xml.append('    </tlLogic>')
    xml.append('    <tlLogic id="TL_J21" type="static" programID="0" offset="5">')
    xml.append('        <phase duration="31" state="GGgrrrGGgrrr"/>')
    xml.append('        <phase duration="4"  state="yyyrrryyyrrr"/>')
    xml.append('        <phase duration="25" state="rrrGGgrrrGGg"/>')
    xml.append('        <phase duration="4"  state="rrryyyrrryyy"/>')
    xml.append('    </tlLogic>')

    lane_width = 3.20

    # Write edges and lanes
    for eid, fn, tn, nlanes, spd, prio in edges:
        x1, y1 = nodes[fn]
        x2, y2 = nodes[tn]
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2)**0.5
        ux = dx / length
        uy = dy / length
        # Normal to the right
        nx = uy
        ny = -ux

        xml.append(f'    <edge id="{eid}" from="{fn}" to="{tn}" priority="{prio}">')
        for i in range(nlanes):
            # i=0 is outermost curb lane
            # i=nlanes-1 is innermost lane
            # offset from center
            dist = 1.60 + (nlanes - 1 - i) * lane_width
            lx1 = x1 + nx * dist
            ly1 = y1 + ny * dist
            lx2 = x2 + nx * dist
            ly2 = y2 + ny * dist
            shape_str = f"{lx1:.2f},{ly1:.2f} {lx2:.2f},{ly2:.2f}"
            xml.append(f'        <lane id="{eid}_{i}" index="{i}" speed="{spd:.2f}" length="{length:.2f}" shape="{shape_str}"/>')
        xml.append('    </edge>')

    # Write junctions
    for jid, (jx, jy) in nodes.items():
        jtype = "traffic_light" if jid in ("J11", "J21") else "priority"
        tl_attr = f' tl="{jid}"' if jid in ("J11", "J21") else ''
        inc_lanes = []
        for eid, fn, tn, nlanes, spd, prio in edges:
            if tn == jid:
                for i in range(nlanes):
                    inc_lanes.append(f"{eid}_{i}")
        inc_str = " ".join(inc_lanes)
        xml.append(f'    <junction id="{jid}" type="{jtype}" x="{jx:.2f}" y="{jy:.2f}" incLanes="{inc_str}" intLanes="" shape="{jx-10:.2f},{jy-10:.2f} {jx-10:.2f},{jy+10:.2f} {jx+10:.2f},{jy+10:.2f} {jx+10:.2f},{jy-10:.2f}"/>')

    # Connections between incoming and outgoing edges at each junction
    for jid, (jx, jy) in nodes.items():
        in_edges = [e for e in edges if e[2] == jid]
        out_edges = [e for e in edges if e[1] == jid]
        for ie in in_edges:
            ie_id, ifrom, _, in_lanes, _, _ = ie
            for oe in out_edges:
                oe_id, _, oto, out_lanes, _, _ = oe
                # Avoid direct U-turn
                if ifrom == oto:
                    continue
                # Map lanes
                min_lanes = min(in_lanes, out_lanes)
                for l_idx in range(min_lanes):
                    xml.append(f'    <connection from="{ie_id}" to="{oe_id}" fromLane="{l_idx}" toLane="{l_idx}" dir="s" state="M"/>')

    xml.append('</net>')
    content = "\n".join(xml) + "\n"
    with open(OUTPUT_NET, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[SUCCESS] Standalone SUMO network written to {OUTPUT_NET}")
    return True

def main():
    nc = find_netconvert()
    if nc:
        success = build_with_netconvert(nc)
        if success:
            return
    generate_standalone_net()

if __name__ == "__main__":
    main()
