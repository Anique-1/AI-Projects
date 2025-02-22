import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Data Classes
@dataclass
class ResourceUsage:
    timestamp: datetime
    fuel_rate: float
    power_consumption: float
    oxygen_level: float
    water_usage: float
    equipment_status: Dict[str, float]

@dataclass
class SupplyItem:
    name: str
    current_level: float
    min_threshold: float
    max_level: float
    unit_cost: float
    priority: int

class SpaceEfficiencyMonitor:
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
    def detect_anomalies(self, data: pd.DataFrame) -> np.ndarray:
        scaled_data = self.scaler.fit_transform(data)
        return self.anomaly_detector.fit_predict(scaled_data)

def generate_mock_data(hours: int = 24) -> pd.DataFrame:
    timestamps = [datetime.now() - timedelta(hours=x) for x in range(hours)]
    
    data = {
        'timestamp': timestamps,
        'fuel_rate': [2.0 + np.random.normal(0, 0.5) for _ in range(hours)],
        'power_consumption': [2.5 + np.random.normal(0, 0.3) for _ in range(hours)],
        'oxygen_level': [95.0 + np.random.normal(0, 2.0) for _ in range(hours)],
        'water_usage': [1.5 + np.random.normal(0, 0.2) for _ in range(hours)],
    }
    
    return pd.DataFrame(data)

def create_line_chart(data: pd.DataFrame, column: str, title: str) -> go.Figure:
    fig = px.line(data, x='timestamp', y=column, title=title)
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=column.replace('_', ' ').title(),
        showlegend=True
    )
    return fig

def create_gauge_chart(value: float, title: str, min_val: float, max_val: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, (max_val-min_val)*0.3], 'color': "red"},
                {'range': [(max_val-min_val)*0.3, (max_val-min_val)*0.7], 'color': "yellow"},
                {'range': [(max_val-min_val)*0.7, max_val], 'color': "green"}
            ]
        }
    ))
    return fig

def main():
    st.set_page_config(page_title="Space Operations Efficiency Monitor", layout="wide")
    
    # Title and description
    st.title("🚀 Space Operations Efficiency Monitor")
    st.markdown("""
    This dashboard monitors and optimizes space operations efficiency in real-time.
    Track resource usage, detect anomalies, and manage supply chain logistics.
    """)
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = generate_mock_data()
    if 'monitor' not in st.session_state:
        st.session_state.monitor = SpaceEfficiencyMonitor()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    if st.sidebar.button("Generate New Data"):
        st.session_state.data = generate_mock_data()
    
    update_frequency = st.sidebar.slider(
        "Update Frequency (seconds)",
        min_value=1,
        max_value=10,
        value=5
    )
    
    # Main dashboard layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Resource Usage Trends")
        
        # Tabs for different metrics
        tab1, tab2, tab3, tab4 = st.tabs([
            "Fuel Rate", 
            "Power Consumption",
            "Oxygen Level",
            "Water Usage"
        ])
        
        with tab1:
            st.plotly_chart(
                create_line_chart(st.session_state.data, 'fuel_rate', 'Fuel Consumption Rate'),
                use_container_width=True
            )
        
        with tab2:
            st.plotly_chart(
                create_line_chart(st.session_state.data, 'power_consumption', 'Power Consumption'),
                use_container_width=True
            )
            
        with tab3:
            st.plotly_chart(
                create_line_chart(st.session_state.data, 'oxygen_level', 'Oxygen Levels'),
                use_container_width=True
            )
            
        with tab4:
            st.plotly_chart(
                create_line_chart(st.session_state.data, 'water_usage', 'Water Usage'),
                use_container_width=True
            )
    
    with col2:
        st.subheader("Current Status")
        
        # Get latest values
        latest = st.session_state.data.iloc[-1]
        
        # Current levels gauges
        st.plotly_chart(
            create_gauge_chart(latest['fuel_rate'], "Fuel Rate", 0, 5),
            use_container_width=True
        )
        
        st.plotly_chart(
            create_gauge_chart(latest['power_consumption'], "Power Consumption", 0, 5),
            use_container_width=True
        )
        
        # Anomaly Detection
        st.subheader("Anomaly Detection")
        
        numerical_data = st.session_state.data.select_dtypes(include=[np.number])
        anomalies = st.session_state.monitor.detect_anomalies(numerical_data)
        anomaly_count = len(anomalies[anomalies == -1])
        
        st.metric(
            "Detected Anomalies",
            anomaly_count,
            delta=None,
            delta_color="inverse"
        )
        
        if anomaly_count > 0:
            st.warning(f"⚠️ {anomaly_count} anomalies detected in the current data")
    
    # Supply Chain Status
    st.subheader("Supply Chain Status")
    
    # Mock supply items
    supply_items = [
        SupplyItem("Fuel", 75.5, 20.0, 100.0, 1000.0, 1),
        SupplyItem("Oxygen", 85.2, 30.0, 100.0, 500.0, 1),
        SupplyItem("Water", 60.8, 25.0, 100.0, 200.0, 2),
        SupplyItem("Food", 45.3, 20.0, 100.0, 300.0, 2)
    ]
    
    supply_cols = st.columns(len(supply_items))
    
    for col, item in zip(supply_cols, supply_items):
        with col:
            st.metric(
                f"{item.name} Level",
                f"{item.current_level:.1f}%",
                delta=f"{item.current_level - item.min_threshold:.1f}% above minimum",
                delta_color="normal"
            )
            
            if item.current_level < item.min_threshold:
                st.error(f"⚠️ {item.name} below minimum threshold!")
            elif item.current_level < item.min_threshold * 1.5:
                st.warning(f"⚡ {item.name} approaching minimum threshold")
    
    # Auto-refresh
    time.sleep(update_frequency)
    st.rerun()

if __name__ == "__main__":
    main()