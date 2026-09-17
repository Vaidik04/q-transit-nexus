import React, { useState, useEffect, useRef } from 'react';
import Navbar from './components/Navbar';
import CityMap from './components/CityMap';
import MetricsGrid from './components/MetricsGrid';
import PredictionPanel from './components/PredictionPanel';
import OptimizerCard from './components/OptimizerCard';
import IncidentControl from './components/IncidentControl';
import ExplainabilityDrawer from './components/ExplainabilityDrawer';
import BenchmarkModal from './components/BenchmarkModal';

const API_BASE = 'http://127.0.0.1:8000';

export default function App() {
  const [simState, setSimState] = useState(null);
  const [metrics, setMetrics] = useState({});
  const [predictions, setPredictions] = useState({});
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [explanations, setExplanations] = useState([]);
  const [selectedVehicle, setSelectedVehicle] = useState(null);

  const [activeMode, setActiveMode] = useState('balanced');
  const [isBenchmarkOpen, setIsBenchmarkOpen] = useState(false);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [isExecutingAction, setIsExecutingAction] = useState(false);

  // Poll simulation state & metrics
  const fetchAllData = async () => {
    try {
      const [stateRes, metricsRes, predRes, optRes, expRes] = await Promise.all([
        fetch(`${API_BASE}/simulation/state`).then(r => r.json()).catch(() => null),
        fetch(`${API_BASE}/metrics`).then(r => r.json()).catch(() => ({})),
        fetch(`${API_BASE}/prediction`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ horizon_minutes: 15 })
        }).then(r => r.json()).catch(() => ({ predictions: {} })),
        fetch(`${API_BASE}/optimize/latest`).then(r => r.json()).catch(() => null),
        fetch(`${API_BASE}/optimize/explanations`).then(r => r.json()).catch(() => [])
      ]);

      if (stateRes) setSimState(stateRes);
      if (metricsRes) setMetrics(metricsRes);
      if (predRes?.predictions) setPredictions(predRes.predictions);
      if (optRes) setOptimizationResult(optRes);
      if (expRes) setExplanations(expRes);
    } catch (err) {
      console.error("Backend connection error", err);
    }
  };

  useEffect(() => {
    fetchAllData();
    const interval = setInterval(fetchAllData, 1500);
    return () => clearInterval(interval);
  }, []);

  // Controls
  const handlePlay = async () => {
    await fetch(`${API_BASE}/simulation/start`, { method: 'POST' });
    fetchAllData();
  };

  const handlePause = async () => {
    await fetch(`${API_BASE}/simulation/stop`, { method: 'POST' });
    fetchAllData();
  };

  const handleStep = async () => {
    await fetch(`${API_BASE}/simulation/step`, { method: 'POST' });
    fetchAllData();
  };

  const handleReset = async () => {
    await fetch(`${API_BASE}/simulation/reset`, { method: 'POST' });
    fetchAllData();
  };

  const handleModeChange = async (mode) => {
    setActiveMode(mode);
    setIsOptimizing(true);
    await fetch(`${API_BASE}/optimize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode, max_iterations: 30 })
    });
    setIsOptimizing(false);
    fetchAllData();
  };

  const handleRunOptimizer = async () => {
    setIsOptimizing(true);
    await fetch(`${API_BASE}/optimize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: activeMode, max_iterations: 35 })
    });
    setIsOptimizing(false);
    fetchAllData();
  };

  const handleSimulateAccident = async (edgeId = 'E14') => {
    setIsExecutingAction(true);
    const res = await fetch(`${API_BASE}/incident`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ edge_id: edgeId, severity: 0.95 })
    }).then(r => r.json()).catch(() => null);
    setIsExecutingAction(false);
    fetchAllData();
    return res;
  };

  const handleClearIncidents = async () => {
    if (simState?.active_incidents?.length > 0) {
      for (const inc of simState.active_incidents) {
        await fetch(`${API_BASE}/incident/clear/${inc.incident_id}`, { method: 'POST' });
      }
      fetchAllData();
    }
  };

  const handleTestTSP = async (busId = 'BUS17', intersection = 'N15') => {
    setIsExecutingAction(true);
    const res = await fetch(`${API_BASE}/optimize/transit-signal?bus_id=${busId}&intersection_id=${intersection}`, {
      method: 'POST'
    }).then(r => r.json()).catch(() => null);
    setIsExecutingAction(false);
    fetchAllData();
    return res;
  };

  const handleTestEVCharger = async (evId = 'EV08') => {
    setIsExecutingAction(true);
    const res = await fetch(`${API_BASE}/optimize/ev-charger?vehicle_id=${evId}`, {
      method: 'POST'
    }).then(r => r.json()).catch(() => null);
    setIsExecutingAction(false);
    fetchAllData();
    return res;
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar
        simState={simState}
        onPlay={handlePlay}
        onPause={handlePause}
        onStep={handleStep}
        onReset={handleReset}
        activeMode={activeMode}
        onModeChange={handleModeChange}
        onOpenBenchmark={() => setIsBenchmarkOpen(true)}
      />

      <main style={{ flex: 1, padding: '16px', maxWidth: '1600px', margin: '0 auto', width: '100%' }}>
        {/* KPI Summary Grid */}
        <MetricsGrid metrics={metrics} />

        {/* Core Operations Section */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1fr) 380px',
          gap: '16px',
          alignItems: 'start'
        }}>
          {/* Left Column: Digital Twin Map & Explainability Card */}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <CityMap
              nodes={simState?.nodes || {}}
              edges={simState?.network?.edges || {}}
              vehicles={simState?.vehicles || []}
              activeIncidents={simState?.active_incidents || []}
              chargerStations={simState?.charger_stations || {}}
              signals={simState?.signals || {}}
              onSelectVehicle={(v) => setSelectedVehicle(v)}
              selectedVehicleId={selectedVehicle?.vehicle_id}
            />

            <ExplainabilityDrawer
              explanations={explanations}
              selectedVehicle={selectedVehicle}
              onClose={() => setSelectedVehicle(null)}
            />
          </div>

          {/* Right Column: Intelligence & Controls */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <IncidentControl
              onSimulateAccident={handleSimulateAccident}
              onClearIncidents={handleClearIncidents}
              onTestTSP={handleTestTSP}
              onTestEVCharger={handleTestEVCharger}
              hasActiveIncidents={(simState?.active_incidents?.length || 0) > 0}
              isExecutingAction={isExecutingAction}
            />

            <PredictionPanel predictions={predictions} />

            <OptimizerCard
              optimizationResult={optimizationResult}
              onRunOptimizer={handleRunOptimizer}
              isOptimizing={isOptimizing}
            />
          </div>
        </div>
      </main>

      {/* Modal for Benchmark, Ablation, Pareto, & Trust Ledger */}
      <BenchmarkModal
        isOpen={isBenchmarkOpen}
        onClose={() => setIsBenchmarkOpen(false)}
      />
    </div>
  );
}
