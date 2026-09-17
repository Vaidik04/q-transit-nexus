import React from 'react';
import { Cpu, RefreshCw, Layers, CheckCircle2 } from 'lucide-react';

export default function OptimizerCard({ optimizationResult, onRunOptimizer, isOptimizing }) {
  const opt = optimizationResult || {
    algorithm: 'AT-DQPSO',
    iterations: 40,
    runtime_sec: 0.28,
    best_fitness: 14.2,
    routes_changed: 3,
    mode_applied: 'balanced'
  };

  return (
    <div className="glass-panel" style={{ padding: '16px', marginBottom: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Cpu size={16} color="#a855f7" />
          <h3 style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '0.5px' }}>
            AT-DQPSO OPTIMIZER
          </h3>
        </div>
        <button
          onClick={onRunOptimizer}
          disabled={isOptimizing}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            padding: '4px 10px',
            borderRadius: '6px',
            border: '1px solid rgba(168, 85, 247, 0.4)',
            background: 'rgba(168, 85, 247, 0.15)',
            color: '#c084fc',
            fontSize: '11px',
            fontWeight: 600
          }}
        >
          <RefreshCw size={12} className={isOptimizing ? 'quantum-ring' : ''} />
          {isOptimizing ? 'Optimizing...' : 'Re-optimize'}
        </button>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, 1fr)',
        gap: '8px',
        fontSize: '11px',
        marginBottom: '12px'
      }}>
        <div style={{ background: 'rgba(30, 41, 59, 0.4)', padding: '8px 10px', borderRadius: '6px' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '10px' }}>Algorithm</div>
          <div style={{ fontWeight: 700, color: '#38bdf8' }}>{opt.algorithm}</div>
        </div>
        <div style={{ background: 'rgba(30, 41, 59, 0.4)', padding: '8px 10px', borderRadius: '6px' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '10px' }}>Best Fitness (F)</div>
          <div style={{ fontWeight: 700, color: '#10b981', fontFamily: 'var(--font-mono)' }}>{opt.best_fitness}</div>
        </div>
        <div style={{ background: 'rgba(30, 41, 59, 0.4)', padding: '8px 10px', borderRadius: '6px' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '10px' }}>Runtime</div>
          <div style={{ fontWeight: 700, color: '#f59e0b', fontFamily: 'var(--font-mono)' }}>{opt.runtime_sec}s</div>
        </div>
        <div style={{ background: 'rgba(30, 41, 59, 0.4)', padding: '8px 10px', borderRadius: '6px' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '10px' }}>Routes Changed</div>
          <div style={{ fontWeight: 700, color: '#ec4899' }}>{opt.routes_changed} routes</div>
        </div>
      </div>

      {/* Fitness Formula Display */}
      <div style={{
        background: 'rgba(15, 23, 42, 0.8)',
        border: '1px solid var(--border-glass)',
        borderRadius: '6px',
        padding: '8px 10px',
        fontSize: '10px',
        color: 'var(--text-secondary)',
        fontFamily: 'var(--font-mono)'
      }}>
        min F = w₁T + w₂D + w₃C + w₄P + w₅E + w₆R + w₇V
      </div>
    </div>
  );
}
