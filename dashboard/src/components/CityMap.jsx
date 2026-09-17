import React, { useState } from 'react';
import { AlertTriangle, Zap, Shield, Navigation } from 'lucide-react';

export default function CityMap({
  nodes = {},
  edges = {},
  vehicles = [],
  activeIncidents = [],
  chargerStations = {},
  signals = {},
  onSelectVehicle,
  selectedVehicleId
}) {
  const [hoveredEntity, setHoveredEntity] = useState(null);

  // Helper to get edge color based on congestion
  const getEdgeColor = (edge) => {
    if (edge?.is_blocked) return '#e11d48'; // Bright crimson
    const cong = edge?.congestion || 0;
    if (cong < 0.35) return '#10b981'; // Smooth green
    if (cong < 0.70) return '#f59e0b'; // Moderate amber
    return '#f43f5e'; // Heavy red
  };

  // Helper to get vehicle icon & color
  const getVehicleMeta = (type) => {
    switch (type) {
      case 'bus':
        return { icon: '🚍', color: '#06b6d4', label: 'Transit Bus' };
      case 'delivery':
        return { icon: '🚚', color: '#6366f1', label: 'Logistics Fleet' };
      case 'ev':
        return { icon: '⚡', color: '#10b981', label: 'Electric Vehicle' };
      case 'emergency':
        return { icon: '🚑', color: '#ef4444', label: 'Emergency Responder' };
      default:
        return { icon: '🚗', color: '#94a3b8', label: 'Private Auto' };
    }
  };

  return (
    <div className="glass-panel" style={{
      position: 'relative',
      height: '100%',
      minHeight: '520px',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Map Header Overlay */}
      <div style={{
        position: 'absolute',
        top: 14,
        left: 16,
        zIndex: 10,
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        background: 'rgba(15, 23, 42, 0.85)',
        padding: '6px 14px',
        borderRadius: '8px',
        border: '1px solid var(--border-glass)'
      }}>
        <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10b981', boxShadow: '0 0 8px #10b981' }} />
        <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)' }}>
          City-Scale Digital Twin Network
        </span>
        <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
          (24 Nodes • 38 Dual Corridors • Live TraCI Feed)
        </span>
      </div>

      {/* Legend Overlay */}
      <div style={{
        position: 'absolute',
        bottom: 14,
        left: 16,
        zIndex: 10,
        display: 'flex',
        gap: '12px',
        background: 'rgba(15, 23, 42, 0.85)',
        padding: '6px 12px',
        borderRadius: '8px',
        border: '1px solid var(--border-glass)',
        fontSize: '11px'
      }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ width: '12px', height: '3px', background: '#10b981', borderRadius: '2px' }} /> Free Flow
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ width: '12px', height: '3px', background: '#f59e0b', borderRadius: '2px' }} /> Moderate
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ width: '12px', height: '3px', background: '#f43f5e', borderRadius: '2px' }} /> Congested
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ width: '12px', height: '3px', background: '#e11d48', borderRadius: '2px' }} /> Blocked Link
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <Zap size={12} color="#10b981" /> EV Charger
        </span>
      </div>

      {/* Interactive SVG Network Canvas */}
      <div style={{ flex: 1, width: '100%', height: '100%', position: 'relative' }}>
        <svg viewBox="50 50 850 540" style={{ width: '100%', height: '100%' }}>
          <defs>
            {/* Glow filters */}
            <filter id="glow-cyan" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <filter id="glow-red" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="5" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Render Network Edges */}
          {Object.values(edges).map((edge) => {
            // Avoid drawing reverse edge twice in exact same pixel
            if (edge.edge_id.endsWith('_R')) return null;

            const u = nodes[edge.from_node];
            const v = nodes[edge.to_node];
            if (!u || !v) return null;

            const isBlocked = edge.is_blocked;
            const color = getEdgeColor(edge);
            const strokeWidth = isBlocked ? 6 : (edge.is_transit_lane ? 4.5 : 3.5);

            return (
              <g key={edge.edge_id} onMouseEnter={() => setHoveredEntity({ type: 'edge', data: edge })}>
                {/* Background road line */}
                <line
                  x1={u.x}
                  y1={u.y}
                  x2={v.x}
                  y2={v.y}
                  stroke="#1e293b"
                  strokeWidth={strokeWidth + 3}
                  strokeLinecap="round"
                />
                {/* Active colored traffic line */}
                <line
                  x1={u.x}
                  y1={u.y}
                  x2={v.x}
                  y2={v.y}
                  stroke={color}
                  strokeWidth={strokeWidth}
                  strokeLinecap="round"
                  strokeDasharray={edge.is_transit_lane ? '6 3' : 'none'}
                  opacity={0.88}
                />
                {/* Edge Label Badge */}
                <text
                  x={(u.x + v.x) / 2}
                  y={(u.y + v.y) / 2 - 5}
                  fill="var(--text-secondary)"
                  fontSize="9"
                  fontFamily="var(--font-mono)"
                  textAnchor="middle"
                  opacity={0.75}
                >
                  {edge.edge_id}
                </text>
              </g>
            );
          })}

          {/* Render Incident Beacons */}
          {activeIncidents.map((inc) => {
            const edge = edges[inc.edge_id];
            if (!edge) return null;
            const u = nodes[edge.from_node];
            const v = nodes[edge.to_node];
            if (!u || !v) return null;
            const mx = (u.x + v.x) / 2;
            const my = (u.y + v.y) / 2;

            return (
              <g key={inc.incident_id} className="incident-beacon">
                <circle cx={mx} cy={my} r="18" fill="rgba(239, 68, 68, 0.35)" />
                <circle cx={mx} cy={my} r="10" fill="#ef4444" />
                <text x={mx} y={my + 4} textAnchor="middle" fontSize="11" fill="#ffffff" fontWeight="bold">
                  🚨
                </text>
                <text x={mx} y={my - 14} textAnchor="middle" fontSize="10" fill="#f87171" fontWeight="bold">
                  {inc.edge_id} ACCIDENT
                </text>
              </g>
            );
          })}

          {/* Render EV Charging Hubs */}
          {Object.values(chargerStations).map((cs) => {
            const n = nodes[cs.node_id];
            if (!n) return null;
            return (
              <g key={cs.station_id} transform={`translate(${n.x + 12}, ${n.y - 12})`}>
                <rect x="-10" y="-10" width="20" height="20" rx="6" fill="#0f172a" stroke="#10b981" strokeWidth="1.5" />
                <text x="0" y="4" textAnchor="middle" fontSize="10" fill="#10b981">⚡</text>
                <text x="0" y="-12" textAnchor="middle" fontSize="8" fill="#a7f3d0" fontWeight="600">
                  Q: {cs.current_queue}
                </text>
              </g>
            );
          })}

          {/* Render Nodes (Intersections) */}
          {Object.entries(nodes).map(([nid, node]) => {
            const sig = signals[nid];
            return (
              <g key={nid}>
                <circle
                  cx={node.x}
                  cy={node.y}
                  r="7"
                  fill="#0d1527"
                  stroke={sig ? '#10b981' : 'rgba(56, 189, 248, 0.4)'}
                  strokeWidth="2"
                  className={sig ? 'signal-active' : ''}
                />
                <text
                  x={node.x}
                  y={node.y + 16}
                  fill="var(--text-secondary)"
                  fontSize="9"
                  fontFamily="var(--font-mono)"
                  textAnchor="middle"
                >
                  {nid}
                </text>
              </g>
            );
          })}

          {/* Render Active Vehicles Moving Live */}
          {vehicles.map((v) => {
            const edge = edges[v.current_edge];
            if (!edge) return null;
            const u = nodes[edge.from_node];
            const toNode = nodes[edge.to_node];
            if (!u || !toNode) return null;

            // Interpolate position along current edge
            const prog = Math.max(0, Math.min(1, v.progress_on_edge || 0));
            const vx = u.x + (toNode.x - u.x) * prog;
            const vy = u.y + (toNode.y - u.y) * prog;

            const meta = getVehicleMeta(v.type);
            const isSelected = selectedVehicleId === v.vehicle_id;

            return (
              <g
                key={v.vehicle_id}
                onClick={() => onSelectVehicle(v)}
                onMouseEnter={() => setHoveredEntity({ type: 'vehicle', data: v })}
                style={{ cursor: 'pointer' }}
              >
                {/* Selection ring */}
                {isSelected && (
                  <circle
                    cx={vx}
                    cy={vy}
                    r="15"
                    fill="none"
                    stroke="#38bdf8"
                    strokeWidth="2"
                    strokeDasharray="4 2"
                    className="quantum-ring"
                  />
                )}

                {/* Vehicle Marker */}
                <circle
                  cx={vx}
                  cy={vy}
                  r={v.type === 'bus' ? 9 : (v.type === 'emergency' ? 8.5 : 7)}
                  fill={meta.color}
                  stroke="#070b14"
                  strokeWidth="2"
                  filter={isSelected ? 'url(#glow-cyan)' : 'none'}
                />

                {/* Icon or Type Symbol */}
                <text
                  x={vx}
                  y={vy + 3.5}
                  textAnchor="middle"
                  fontSize={v.type === 'bus' ? '9' : '8'}
                  fill="#ffffff"
                  fontWeight="bold"
                >
                  {meta.icon}
                </text>

                {/* ID Label & Badge */}
                <text
                  x={vx}
                  y={vy - 10}
                  textAnchor="middle"
                  fontSize="8"
                  fill="#f8fafc"
                  fontFamily="var(--font-mono)"
                  fontWeight="600"
                >
                  {v.vehicle_id}
                </text>

                {/* Bus Passenger Badge */}
                {v.type === 'bus' && (
                  <g transform={`translate(${vx + 8}, ${vy - 8})`}>
                    <rect x="-8" y="-6" width="16" height="11" rx="3" fill="#0369a1" />
                    <text x="0" y="3" textAnchor="middle" fontSize="7" fill="#ffffff" fontWeight="bold">
                      {v.passengers}p
                    </text>
                  </g>
                )}
              </g>
            );
          })}
        </svg>

        {/* Hover Tooltip Details */}
        {hoveredEntity && (
          <div style={{
            position: 'absolute',
            bottom: 46,
            right: 16,
            background: 'rgba(15, 23, 42, 0.95)',
            border: '1px solid var(--border-glass-bright)',
            borderRadius: '8px',
            padding: '10px 14px',
            fontSize: '11px',
            pointerEvents: 'none',
            zIndex: 30,
            maxWidth: '260px'
          }}>
            {hoveredEntity.type === 'vehicle' ? (
              <div>
                <div style={{ fontWeight: 700, color: '#38bdf8', marginBottom: '4px' }}>
                  {hoveredEntity.data.vehicle_id} ({hoveredEntity.data.type.toUpperCase()})
                </div>
                <div>Speed: {hoveredEntity.data.current_speed} km/h</div>
                <div>Edge: {hoveredEntity.data.current_edge}</div>
                <div>Trip: {hoveredEntity.data.origin} → {hoveredEntity.data.destination}</div>
                {hoveredEntity.data.passengers > 0 && <div>Passengers: {hoveredEntity.data.passengers}</div>}
                {hoveredEntity.data.soc !== undefined && <div>Battery SOC: {Math.round(hoveredEntity.data.soc * 100)}%</div>}
              </div>
            ) : (
              <div>
                <div style={{ fontWeight: 700, color: '#38bdf8', marginBottom: '4px' }}>
                  Road {hoveredEntity.data.edge_id}
                </div>
                <div>Speed: {hoveredEntity.data.speed} km/h (Limit: {hoveredEntity.data.free_flow_speed})</div>
                <div>Travel Time: {hoveredEntity.data.travel_time}s</div>
                <div>Congestion: {Math.round(hoveredEntity.data.congestion * 100)}%</div>
                <div>Capacity: {hoveredEntity.data.capacity} vph</div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
