import React, { useState } from 'react';
import { TrendingUp, Clock, AlertTriangle } from 'lucide-react';

export default function PredictionPanel({ predictions = {} }) {
  const [horizon, setHorizon] = useState('t10'); // t5, t10, t15

  const predList = Object.values(predictions).slice(0, 5);

  return (
    <div className="glass-panel" style={{ padding: '16px', marginBottom: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <TrendingUp size={16} color="#06b6d4" />
          <h3 style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '0.5px' }}>
            ML / SPATIO-TEMPORAL PREDICTOR
          </h3>
        </div>
        {/* Horizon Tabs */}
        <div style={{ display: 'flex', gap: '4px', background: 'rgba(15, 23, 42, 0.8)', padding: '2px', borderRadius: '6px' }}>
          {['t5', 't10', 't15'].map(h => (
            <button
              key={h}
              onClick={() => setHorizon(h)}
              style={{
                padding: '3px 8px',
                border: 'none',
                borderRadius: '4px',
                fontSize: '10px',
                fontWeight: 700,
                background: horizon === h ? 'rgba(6, 182, 212, 0.3)' : 'transparent',
                color: horizon === h ? '#38bdf8' : 'var(--text-muted)'
              }}
            >
              {h === 't5' ? 'T+5m' : (h === 't10' ? 'T+10m' : 'T+15m')}
            </button>
          ))}
        </div>
      </div>

      <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '10px' }}>
        Predicting downstream edge conditions before vehicles encounter bottlenecks.
      </p>

      {/* Critical Edges Prediction List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {predList.map((p) => {
          const tt = horizon === 't5' ? p.t5 : (horizon === 't10' ? p.t10 : p.t15);
          const spd = horizon === 't5' ? p.speed_t5 : (horizon === 't10' ? p.speed_t10 : p.speed_t15);
          const cong = horizon === 't5' ? p.congestion_t5 : (horizon === 't10' ? p.congestion_t10 : p.congestion_t15);
          const isBottleneck = (cong || 0) > 0.65;

          return (
            <div key={p.edge_id} style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '8px 10px',
              borderRadius: '6px',
              background: isBottleneck ? 'rgba(239, 68, 68, 0.1)' : 'rgba(30, 41, 59, 0.5)',
              border: isBottleneck ? '1px solid rgba(239, 68, 68, 0.3)' : '1px solid rgba(255, 255, 255, 0.05)',
              fontSize: '11px'
            }}>
              <div>
                <div style={{ fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span>{p.edge_id}</span>
                  {isBottleneck && <span style={{ color: '#ef4444', fontSize: '9px', fontWeight: 700 }}>HIGH RISK</span>}
                </div>
                <div style={{ color: 'var(--text-muted)', fontSize: '10px' }}>
                  Pred Speed: {spd} km/h • P95: {p.p95_travel_time}s
                </div>
              </div>

              <div style={{ textAlign: 'right' }}>
                <div style={{ fontWeight: 700, color: isBottleneck ? '#f87171' : '#38bdf8', fontFamily: 'var(--font-mono)' }}>
                  {tt}s
                </div>
                <div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>
                  Congestion: {Math.round((cong || 0.2) * 100)}%
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
