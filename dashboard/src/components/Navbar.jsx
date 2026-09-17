import React from 'react';
import { Play, Pause, SkipForward, RotateCcw, Activity, ShieldAlert, Cpu, BarChart2 } from 'lucide-react';

export default function Navbar({
  simState,
  onPlay,
  onPause,
  onStep,
  onReset,
  activeMode,
  onModeChange,
  onOpenBenchmark,
  onOpenPareto
}) {
  const isRunning = simState?.is_running;
  const modes = [
    { id: 'balanced', label: 'Balanced City', icon: '⚖️' },
    { id: 'emergency', label: 'Emergency Priority', icon: '🚨' },
    { id: 'green', label: 'Green Eco-Mode', icon: '🌱' },
    { id: 'transit', label: 'Public Transit', icon: '🚍' },
  ];

  return (
    <header style={{
      height: '72px',
      padding: '0 24px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      background: 'rgba(13, 21, 39, 0.85)',
      backdropFilter: 'blur(20px)',
      borderBottom: '1px solid var(--border-glass)',
      position: 'sticky',
      top: 0,
      zIndex: 100
    }}>
      {/* Brand & Subtitle */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div style={{
          width: '42px',
          height: '42px',
          borderRadius: '10px',
          background: 'linear-gradient(135deg, #06b6d4, #6366f1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 0 15px rgba(6, 182, 212, 0.5)'
        }}>
          <Cpu size={24} color="#ffffff" />
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <h1 style={{
              fontFamily: 'var(--font-display)',
              fontSize: '20px',
              fontWeight: 800,
              letterSpacing: '1px',
              background: 'linear-gradient(90deg, #f8fafc, #38bdf8)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent'
            }}>
              Q-TRANSIT NEXUS
            </h1>
            <span style={{
              fontSize: '10px',
              fontWeight: 700,
              padding: '2px 8px',
              borderRadius: '20px',
              background: 'rgba(6, 182, 212, 0.15)',
              color: '#38bdf8',
              border: '1px solid rgba(56, 189, 248, 0.3)',
              fontFamily: 'var(--font-mono)'
            }}>
              AT-DQPSO TWIN
            </span>
          </div>
          <p style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '2px' }}>
            Predictive Multimodal Transportation Digital Twin
          </p>
        </div>
      </div>

      {/* Mode Switcher */}
      <div style={{
        display: 'flex',
        background: 'rgba(15, 23, 42, 0.8)',
        padding: '4px',
        borderRadius: '10px',
        border: '1px solid var(--border-glass)'
      }}>
        {modes.map(m => (
          <button
            key={m.id}
            onClick={() => onModeChange(m.id)}
            style={{
              padding: '6px 14px',
              borderRadius: '7px',
              border: 'none',
              fontSize: '12px',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: activeMode === m.id ? 'linear-gradient(135deg, rgba(6, 182, 212, 0.25), rgba(99, 102, 241, 0.25))' : 'transparent',
              color: activeMode === m.id ? '#38bdf8' : 'var(--text-secondary)',
              boxShadow: activeMode === m.id ? '0 0 10px rgba(56, 189, 248, 0.2)' : 'none',
              border: activeMode === m.id ? '1px solid rgba(56, 189, 248, 0.4)' : '1px solid transparent'
            }}
          >
            <span>{m.icon}</span>
            {m.label}
          </button>
        ))}
      </div>

      {/* Simulation Stepper Controls & Modal Launchers */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <button
          onClick={isRunning ? onPause : onPlay}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '8px 16px',
            borderRadius: '8px',
            border: 'none',
            fontSize: '13px',
            fontWeight: 600,
            background: isRunning ? 'rgba(239, 68, 68, 0.2)' : 'linear-gradient(135deg, #06b6d4, #2563eb)',
            color: isRunning ? '#f87171' : '#ffffff',
            border: isRunning ? '1px solid rgba(239, 68, 68, 0.4)' : 'none',
            boxShadow: isRunning ? '0 0 12px rgba(239, 68, 68, 0.3)' : '0 0 15px rgba(6, 182, 212, 0.4)'
          }}
        >
          {isRunning ? <Pause size={16} /> : <Play size={16} />}
          {isRunning ? 'Pause Sim' : 'Live Twin'}
        </button>

        <button
          onClick={onStep}
          title="Advance 1 Step"
          style={{
            padding: '8px 12px',
            borderRadius: '8px',
            background: 'rgba(30, 41, 59, 0.8)',
            border: '1px solid var(--border-glass)',
            color: 'var(--text-primary)',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            fontSize: '12px'
          }}
        >
          <SkipForward size={15} />
          Step
        </button>

        <button
          onClick={onReset}
          title="Reset Simulation"
          style={{
            padding: '8px 10px',
            borderRadius: '8px',
            background: 'rgba(30, 41, 59, 0.8)',
            border: '1px solid var(--border-glass)',
            color: 'var(--text-secondary)'
          }}
        >
          <RotateCcw size={15} />
        </button>

        <div style={{ height: '24px', width: '1px', background: 'rgba(255, 255, 255, 0.15)', margin: '0 4px' }} />

        <button
          onClick={onOpenBenchmark}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '8px 14px',
            borderRadius: '8px',
            background: 'rgba(99, 102, 241, 0.15)',
            border: '1px solid rgba(99, 102, 241, 0.4)',
            color: '#a5b4fc',
            fontSize: '12px',
            fontWeight: 600
          }}
        >
          <BarChart2 size={16} />
          Benchmarks & Ablation
        </button>
      </div>
    </header>
  );
}
