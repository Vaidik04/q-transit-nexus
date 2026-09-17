import React, { useState } from 'react';
import { AlertOctagon, CheckCircle2, Zap, Radio } from 'lucide-react';

export default function IncidentControl({
  onSimulateAccident,
  onClearIncidents,
  onTestTSP,
  onTestEVCharger,
  hasActiveIncidents,
  isExecutingAction
}) {
  const [lastActionLog, setLastActionLog] = useState(null);

  const handleSimulate = async () => {
    const res = await onSimulateAccident('E14');
    setLastActionLog({
      type: 'accident',
      title: '🚨 ROAD E14 ACCIDENT INJECTED',
      details: 'GNN predicted downstream congestion spike. AT-DQPSO automatically triggered closed-loop re-optimization: diverted delivery routes, reserved Bus 17 corridor, and updated signal timing.'
    });
  };

  const handleTSP = async () => {
    const res = await onTestTSP('BUS17', 'N15');
    setLastActionLog({
      type: 'tsp',
      title: '🚍 TRANSIT SIGNAL PRIORITY GRANTED (BUS 17)',
      details: '82 passengers approaching N15. Priority granted: Green phase extended 15s. Societal delay saving: 61.5 passenger-minutes vs +14s cross-traffic cost.'
    });
  };

  const handleEV = async () => {
    const res = await onTestEVCharger('EV08');
    setLastActionLog({
      type: 'ev',
      title: '⚡ EV 08 CHARGING CO-OPTIMIZATION',
      details: 'Diverted from nearest station CS_N21 (25m queue) to CS_N10 (4m queue). Total trip + charging time saved: 18.2 minutes.'
    });
  };

  return (
    <div className="glass-panel" style={{ padding: '16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
        <AlertOctagon size={16} color="#ef4444" />
        <h3 style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '0.5px' }}>
          INCIDENT & ADAPTIVE CONTROL
        </h3>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {/* Main Demo Action Button */}
        <button
          onClick={handleSimulate}
          disabled={isExecutingAction}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            padding: '12px',
            borderRadius: '8px',
            border: 'none',
            background: 'linear-gradient(135deg, #ef4444, #b91c1c)',
            color: '#ffffff',
            fontSize: '13px',
            fontWeight: 800,
            letterSpacing: '0.5px',
            boxShadow: '0 0 20px rgba(239, 68, 68, 0.4)',
            cursor: 'pointer'
          }}
        >
          <AlertOctagon size={18} />
          SIMULATE ACCIDENT ON ROAD 14
        </button>

        {hasActiveIncidents && (
          <button
            onClick={onClearIncidents}
            style={{
              padding: '8px',
              borderRadius: '6px',
              border: '1px solid rgba(16, 185, 129, 0.4)',
              background: 'rgba(16, 185, 129, 0.15)',
              color: '#34d399',
              fontSize: '11px',
              fontWeight: 600
            }}
          >
            Clear Active Incidents & Restore Normal Flow
          </button>
        )}

        {/* Additional Demonstration Triggers */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginTop: '4px' }}>
          <button
            onClick={handleTSP}
            disabled={isExecutingAction}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              padding: '8px 10px',
              borderRadius: '6px',
              border: '1px solid rgba(6, 182, 212, 0.3)',
              background: 'rgba(6, 182, 212, 0.12)',
              color: '#38bdf8',
              fontSize: '11px',
              fontWeight: 600
            }}
          >
            <Radio size={14} />
            TSP (Bus 17)
          </button>

          <button
            onClick={handleEV}
            disabled={isExecutingAction}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              padding: '8px 10px',
              borderRadius: '6px',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              background: 'rgba(16, 185, 129, 0.12)',
              color: '#34d399',
              fontSize: '11px',
              fontWeight: 600
            }}
          >
            <Zap size={14} />
            EV Smart Charge
          </button>
        </div>

        {/* Cascade Feedback Banner */}
        {lastActionLog && (
          <div style={{
            marginTop: '8px',
            padding: '10px 12px',
            borderRadius: '6px',
            background: 'rgba(15, 23, 42, 0.9)',
            border: '1px solid var(--border-glass-bright)',
            fontSize: '11px'
          }}>
            <div style={{ fontWeight: 700, color: '#38bdf8', marginBottom: '4px' }}>
              {lastActionLog.title}
            </div>
            <div style={{ color: 'var(--text-secondary)', lineHeight: '1.4' }}>
              {lastActionLog.details}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
