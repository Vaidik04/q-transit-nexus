import requests
import json

BASE = 'http://127.0.0.1:8000'

print('1. Testing /health...')
r = requests.get(f'{BASE}/health')
assert r.status_code == 200, f'Health failed: {r.text}'
print('  OK:', r.json())

print('2. Testing /simulation/state...')
r = requests.get(f'{BASE}/simulation/state')
assert r.status_code == 200
data = r.json()
print(f'  OK: Nodes: {len(data["nodes"])}, Edges: {len(data["network"]["edges"])}, Vehicles: {len(data["vehicles"])}')

print('3. Testing /vehicles & /traffic/current...')
r_veh = requests.get(f'{BASE}/vehicles')
assert r_veh.status_code == 200
r_traf = requests.get(f'{BASE}/traffic/current')
assert r_traf.status_code == 200
print(f'  OK: /vehicles returned {len(r_veh.json())} vehicles. /traffic/current active vehicles: {r_traf.json().get("total_vehicles")}')

print('4. Testing /prediction (T+5, T+10, T+15)...')
r = requests.post(f'{BASE}/prediction', json={'horizon_minutes': 15})
assert r.status_code == 200
preds = r.json()['predictions']
print(f'  OK: Predictions received for {len(preds)} edges. E14: {preds.get("E14")}')

print('5. Testing [SIMULATE ACCIDENT] /incident on E14...')
r = requests.post(f'{BASE}/incident', json={'edge_id': 'E14', 'severity': 0.95})
assert r.status_code == 200
inc = r.json()
print('  OK:', inc['message'])

print('6. Testing /optimize (AT-DQPSO multi-objective)...')
r = requests.post(f'{BASE}/optimize', json={'mode': 'emergency', 'max_iterations': 30})
assert r.status_code == 200
opt = r.json()
print(f'  OK: Algorithm: {opt["algorithm"]}, Runtime: {opt["runtime_sec"]}s, Fitness: {opt["best_fitness"]}, Routes Changed: {opt["routes_changed"]}')

print('7. Testing /optimize/explain/V27...')
r = requests.get(f'{BASE}/optimize/explain/V27')
assert r.status_code == 200
exp = r.json()
print(f'  OK: V27 Rationale: {exp.get("decision_rationale")}, Expected Savings: {exp.get("expected_saving_min")} min')

print('8. Testing /metrics...')
r = requests.get(f'{BASE}/metrics')
assert r.status_code == 200
met = r.json()
print('  OK: Metrics: Traffic:', met.get("traffic"), 'Congestion:', met.get("congestion"), 'Vehicles:', met.get("vehicle_count"), 'Speed:', met.get("average_speed"), 'CO2:', met.get("co2"))

print('9. Testing /simulation/start & /simulation/stop...')
r_start = requests.post(f'{BASE}/simulation/start')
assert r_start.status_code == 200 and r_start.json().get('status') == 'started'
r_stop = requests.post(f'{BASE}/simulation/stop')
assert r_stop.status_code == 200 and r_stop.json().get('status') == 'stopped'
print('  OK: Simulation start and stop control verified.')

print('10. Testing /benchmark...')
r = requests.get(f'{BASE}/benchmark')
assert r.status_code == 200
bench = r.json()
print(f'  OK: Benchmark evaluated {len(bench["results"])} algorithms (Source: {bench.get("source")}):')
for res in bench['results']:
    print(f'     {res["algorithm"]:<16} | Time: {res.get("travel_time_sec")}s | Runtime: {res.get("runtime_sec")}s | Violations: {res.get("violations")}')

print('11. Testing /ablation...')
r_abl = requests.get(f'{BASE}/ablation')
assert r_abl.status_code == 200
abl = r_abl.json()
print(f'  OK: Ablation evaluated {len(abl.get("configurations", []))} configurations (Source: {abl.get("source")}):')
for cfg in abl.get("configurations", []):
    print(f'     {cfg.get("configuration", cfg.get("variant")):<22} | Time: {cfg.get("travel_time_sec")}s | Congestion: {cfg.get("network_congestion")} | Obj: {cfg.get("objective")}')

print('\nALL PIPELINE CHECKS PASSED PERFECTLY!')
