import React from 'react';
import { Activity, Gauge, Clock, Users, Leaf, AlertCircle } from 'lucide-react';

export default function MetricsGrid({ metrics = {} }) {
  const cards = [
    {
      title: 'Avg Congestion',
      value: `${Math.round((metrics.congestion_index || 0.2) * 100)}%`,
      icon: <Activity size={18} color="#06b6d4" />,
      sub: `${metrics.traffic_density || '20%'} of city capacity`,
      color: metrics.congestion_index > 0.6 ? '#f43f5e' : (metrics.congestion_index > 0.35 ? '#f59e0b' : '#10b981')
    },
    {
      title: 'Average Speed',
      value: `${metrics.average_speed_kmh || 38.4} km/h`,
      icon: <Gauge size={18} color="#3b82f6" />,
      sub: 'Free-flow benchmark 48 km/h',
      color: '#38bdf8'
    },
    {
      title: 'Bus Avg Delay',
      value: `+${metrics.bus_average_delay_min || 1.4} min`,
      icon: <Clock size={18} color="#f59e0b" />,
      sub: 'GTFS-RT Schedule offset',
      color: '#fbbf24'
    },
    {
      title: 'Passenger Delay',
      value: `${Math.round(metrics.passenger_delay_person_min || 142)}`,
      unit: 'person-min',
      icon: <Users size={18} color="#a855f7" />,
      sub: 'Weighted societal delay',
      color: '#c084fc'
    },
    {
      title: 'CO₂ Reduction',
      value: `↓ ${metrics.co2_savings_percent || 12.4}%`,
      icon: <Leaf size={18} color="#10b981" />,
      sub: `${metrics.total_co2_kg || 4.2} kg CO₂ active`,
      color: '#34d399'
    },
    {
      title: 'Active Incidents',
      value: `${metrics.active_incidents || 0}`,
      icon: <AlertCircle size={18} color={metrics.active_incidents > 0 ? '#ef4444' : '#64748b'} />,
      sub: metrics.active_incidents > 0 ? 'Rerouting active' : 'Network clear',
      color: metrics.active_incidents > 0 ? '#f87171' : '#94a3b8'
    },
  ];

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(6, 1fr)',
      gap: '12px',
      marginBottom: '16px'
    }}>
      {cards.map((c, i) => (
        <div key={i} className="glass-panel" style={{ padding: '12px 14px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-secondary)', fontWeight: 500 }}>{c.title}</span>
            {c.icon}
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px' }}>
            <span style={{
              fontSize: '20px',
              fontFamily: 'var(--font-display)',
              fontWeight: 700,
              color: c.color
            }}>
              {c.value}
            </span>
            {c.unit && <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{c.unit}</span>}
          </div>
          <p style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>
            {c.sub}
          </p>
        </div>
      ))}
    </div>
  );
}
