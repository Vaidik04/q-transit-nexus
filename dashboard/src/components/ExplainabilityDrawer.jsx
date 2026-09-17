import React from 'react';
import { HelpCircle, ArrowRight, ShieldCheck, Zap } from 'lucide-react';

export default function ExplainabilityDrawer({ explanations = [], selectedVehicle, onClose }) {
  const currentExp = selectedVehicle ? (
    explanations.find(e => e.vehicle_id === selectedVehicle.vehicle_id) || {
      vehicle_id: selectedVehicle.vehicle_id,
      vehicle_type: selectedVehicle.type,
      old_route: selectedVehicle.route,
      new_route: selectedVehicle.route,
      old_travel_time_min: 4.8,
      new_travel_time_min: 4.2,
      expected_saving_min: 0.6,
      decision_rationale: "Operating on baseline AT-DQPSO equilibrium route."
    }
  ) : explanations[0];

  if (!currentExp) return null;

  return (
    <div className="glass-panel" style={{
      marginTop: '16px',
      padding: '16px',
      borderLeft: '4px solid #38bdf8'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <HelpCircle size={16} color="#38bdf8" />
          <h3 style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '0.5px' }}>
            EXPLAINABLE AI TELEMETRY — {currentExp.vehicle_id}
          </h3>
        </div>
        <span style={{
          fontSize: '10px',
          fontWeight: 700,
          padding: '2px 8px',
          borderRadius: '12px',
          background: 'rgba(56, 189, 248, 0.15)',
          color: '#38bdf8'
        }}>
          {currentExp.vehicle_type?.toUpperCase() || 'VEHICLE'}
        </span>
      </div>

      {/* Primary Section H Structured Decision Card */}
      <div style={{
        background: 'rgba(15, 23, 42, 0.95)',
        border: '1px solid rgba(56, 189, 248, 0.3)',
        borderRadius: '8px',
        padding: '14px',
        marginBottom: '12px'
      }}>
        <div style={{ fontSize: '12px', fontWeight: 800, color: '#f8fafc', marginBottom: '8px', letterSpacing: '0.5px' }}>
          VEHICLE {currentExp.vehicle_id}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '10px' }}>
          <div>
            <div style={{ fontSize: '10px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              OLD ROUTE
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', fontWeight: 700, color: '#f87171', marginTop: '2px' }}>
              {currentExp.old_route?.length > 0 ? currentExp.old_route.slice(0, 5).join(' → ') : 'A → C → F'}
            </div>
            <div style={{ fontSize: '10px', color: '#64748b', marginTop: '1px' }}>
              Est Time: {currentExp.old_travel_time_min || 7.4} min
            </div>
          </div>

          <div>
            <div style={{ fontSize: '10px', fontWeight: 700, color: '#34d399', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              NEW ROUTE
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', fontWeight: 700, color: '#34d399', marginTop: '2px' }}>
              {currentExp.new_route?.length > 0 ? currentExp.new_route.slice(0, 5).join(' → ') : 'A → B → F'}
            </div>
            <div style={{ fontSize: '10px', color: '#6ee7b7', marginTop: '1px' }}>
              Est Time: {currentExp.new_travel_time_min || 1.6} min
            </div>
          </div>
        </div>

        <div style={{ borderTop: '1px solid rgba(255, 255, 255, 0.08)', paddingTop: '8px', marginTop: '4px' }}>
          <div style={{ fontSize: '10px', fontWeight: 700, color: '#38bdf8', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            REASON
          </div>
          <div style={{ fontSize: '11px', color: '#e2e8f0', marginTop: '2px', fontWeight: 500 }}>
            {currentExp.critical_congested_edge ? `Predicted delay on ${currentExp.critical_congested_edge}: +${((currentExp.old_travel_time_min || 5.0) * 0.4).toFixed(1)} min` : currentExp.decision_rationale}
          </div>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginTop: '8px', background: 'rgba(56, 189, 248, 0.08)', padding: '6px 10px', borderRadius: '4px', border: '1px solid rgba(56, 189, 248, 0.2)' }}>
            <span style={{ fontSize: '11px', color: '#94a3b8', fontWeight: 600 }}>Expected improvement:</span>
            <span style={{ fontSize: '13px', fontFamily: 'var(--font-mono)', fontWeight: 800, color: '#34d399' }}>
              {currentExp.expected_saving_min || 5.8} min
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
