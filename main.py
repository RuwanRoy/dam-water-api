"""
Dam Water Level Simulation API
Built with FastAPI - Simulates dam water levels based on rainfall
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import List, Optional
import random
import math

# Initialize FastAPI app
app = FastAPI(
    title="Dam Water Level Simulation API",
    description="API for simulating dam water levels based on rainfall data",
    version="1.0.0"
)

# Enable CORS (so frontend can connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Data Models ====================

class DamInfo(BaseModel):
    """Dam Information Model"""
    dam_id: str
    name: str
    location: str
    capacity: float = Field(..., description="Total capacity in cubic meters")
    max_level: float = Field(..., description="Maximum water level in meters")
    catchment_area: float = Field(..., description="Catchment area in sq km")
    current_level: float = Field(..., description="Current water level percentage")
    status: str

class RainfallData(BaseModel):
    """Rainfall Data Model"""
    timestamp: datetime
    rainfall_mm: float = Field(..., description="Rainfall in millimeters")
    location: str

class WaterLevelResponse(BaseModel):
    """Water Level Response Model"""
    timestamp: datetime
    level_percentage: float
    level_meters: float
    storage_volume: float
    status: str
    inflow_rate: float
    outflow_rate: float

class HistoricalDataPoint(BaseModel):
    """Historical Data Point"""
    timestamp: str
    level: float
    rainfall: float
    inflow: float
    outflow: float
    storage: float

class SimulationRequest(BaseModel):
    """Simulation Request Model"""
    rainfall_mm: float = Field(..., ge=0, le=500, description="Rainfall in mm")
    duration_hours: int = Field(default=24, ge=1, le=168, description="Simulation duration")
    initial_level: float = Field(default=65.0, ge=0, le=100, description="Initial water level %")

# ==================== In-Memory Database (Simulation) ====================

class DamSimulator:
    """Dam Water Level Simulator"""
    
    def __init__(self):
        self.dam_data = {
            "dam_id": "DAM001",
            "name": "Mullaperiyar Dam",
            "location": "Kerala, India",
            "capacity": 15000000,  # 15 million cubic meters
            "max_level": 142,  # feet (converted to 43.28 meters)
            "catchment_area": 624,  # sq km
            "current_level": 65.0,  # percentage
            "base_inflow": 50.0,  # cubic meters per second
            "base_outflow": 45.0,  # cubic meters per second
        }
        self.historical_data = []
        self.last_update = datetime.now()
    
    def calculate_inflow(self, rainfall_mm: float, catchment_area: float) -> float:
        """
        Calculate inflow based on rainfall
        Formula: Inflow = (Rainfall * Catchment Area * Runoff Coefficient) / Time
        """
        runoff_coefficient = 0.7  # 70% of rainfall becomes runoff
        # Convert mm to meters, multiply by area in sq meters
        rainfall_m = rainfall_mm / 1000
        area_sq_m = catchment_area * 1_000_000
        
        # Calculate volume in cubic meters
        volume = rainfall_m * area_sq_m * runoff_coefficient
        
        # Convert to flow rate (m³/s) - assume rainfall over 1 hour
        flow_rate = volume / 3600  # cubic meters per second
        
        return self.dam_data["base_inflow"] + flow_rate
    
    def calculate_outflow(self, current_level: float) -> float:
        """Calculate outflow based on current water level"""
        base_outflow = self.dam_data["base_outflow"]
        
        # If level is high, increase outflow to prevent overflow
        if current_level > 90:
            return base_outflow * 1.8
        elif current_level > 75:
            return base_outflow * 1.4
        elif current_level < 30:
            return base_outflow * 0.6
        
        return base_outflow
    
    def update_water_level(self, rainfall_mm: float = 0) -> WaterLevelResponse:
        """Update water level based on rainfall and flows"""
        # Calculate flows
        inflow = self.calculate_inflow(rainfall_mm, self.dam_data["catchment_area"])
        outflow = self.calculate_outflow(self.dam_data["current_level"])
        
        # Net change in volume per second
        net_flow = inflow - outflow
        
        # Time difference (assume 1 hour for simulation)
        time_delta_seconds = 3600
        
        # Calculate volume change
        volume_change = net_flow * time_delta_seconds
        
        # Convert to percentage change
        capacity = self.dam_data["capacity"]
        percentage_change = (volume_change / capacity) * 100
        
        # Update current level
        new_level = self.dam_data["current_level"] + percentage_change
        new_level = max(0, min(100, new_level))  # Clamp between 0-100
        
        self.dam_data["current_level"] = new_level
        self.last_update = datetime.now()
        
        # Determine status
        if new_level >= 90:
            status = "CRITICAL"
        elif new_level >= 75:
            status = "WARNING"
        elif new_level >= 40:
            status = "NORMAL"
        else:
            status = "LOW"
        
        # Calculate actual level in meters
        level_meters = (new_level / 100) * 43.28  # Max level in meters
        storage_volume = (new_level / 100) * capacity
        
        return WaterLevelResponse(
            timestamp=self.last_update,
            level_percentage=round(new_level, 2),
            level_meters=round(level_meters, 2),
            storage_volume=round(storage_volume, 2),
            status=status,
            inflow_rate=round(inflow, 2),
            outflow_rate=round(outflow, 2)
        )
    
    def get_dam_info(self) -> DamInfo:
        """Get current dam information"""
        status = "NORMAL"
        level = self.dam_data["current_level"]
        
        if level >= 90:
            status = "CRITICAL"
        elif level >= 75:
            status = "WARNING"
        elif level < 40:
            status = "LOW"
        
        return DamInfo(
            dam_id=self.dam_data["dam_id"],
            name=self.dam_data["name"],
            location=self.dam_data["location"],
            capacity=self.dam_data["capacity"],
            max_level=self.dam_data["max_level"],
            catchment_area=self.dam_data["catchment_area"],
            current_level=round(self.dam_data["current_level"], 2),
            status=status
        )
    
    def generate_historical_data(self, hours: int = 24) -> List[HistoricalDataPoint]:
        """Generate historical data for visualization"""
        data = []
        now = datetime.now()
        
        for i in range(hours, -1, -1):
            timestamp = now - timedelta(hours=i)
            
            # Simulate varying conditions
            base_level = 60
            level_variation = 15 * math.sin(i * 0.3) + random.uniform(-3, 3)
            level = max(20, min(95, base_level + level_variation))
            
            rainfall = max(0, random.gauss(5, 10))  # Random rainfall
            inflow = 50 + rainfall * 2 + random.uniform(-5, 5)
            outflow = 45 + random.uniform(-5, 5)
            storage = (level / 100) * self.dam_data["capacity"]
            
            data.append(HistoricalDataPoint(
                timestamp=timestamp.strftime("%H:%M"),
                level=round(level, 2),
                rainfall=round(rainfall, 2),
                inflow=round(inflow, 2),
                outflow=round(outflow, 2),
                storage=round(storage, 2)
            ))
        
        return data

# Initialize simulator
simulator = DamSimulator()

# ==================== API Endpoints ====================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Dam Water Level Simulation API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "dam_info": "/api/dam/info",
            "current_level": "/api/water-level/current",
            "update_level": "/api/water-level/update",
            "historical": "/api/water-level/historical",
            "simulate": "/api/simulate"
        }
    }

@app.get("/api/dam/info", response_model=DamInfo, tags=["Dam Information"])
async def get_dam_info():
    """
    Get current dam information
    
    Returns:
    - Dam ID, name, location
    - Capacity and current level
    - Status
    """
    return simulator.get_dam_info()

@app.get("/api/water-level/current", response_model=WaterLevelResponse, tags=["Water Level"])
async def get_current_water_level():
    """
    Get current water level data
    
    Returns:
    - Current water level (percentage and meters)
    - Storage volume
    - Inflow and outflow rates
    - Status
    """
    return simulator.update_water_level(rainfall_mm=0)

@app.post("/api/water-level/update", response_model=WaterLevelResponse, tags=["Water Level"])
async def update_water_level(rainfall_mm: float = Query(0, ge=0, le=500, description="Rainfall in millimeters")):
    """
    Update water level based on rainfall
    
    Parameters:
    - rainfall_mm: Rainfall amount in millimeters (0-500)
    
    Returns:
    - Updated water level data
    """
    return simulator.update_water_level(rainfall_mm=rainfall_mm)

@app.get("/api/water-level/historical", response_model=List[HistoricalDataPoint], tags=["Water Level"])
async def get_historical_data(hours: int = Query(24, ge=1, le=168, description="Number of hours")):
    """
    Get historical water level data
    
    Parameters:
    - hours: Number of hours of historical data (1-168)
    
    Returns:
    - List of historical data points
    """
    return simulator.generate_historical_data(hours=hours)

@app.post("/api/simulate", tags=["Simulation"])
async def run_simulation(request: SimulationRequest):
    """
    Run a complete simulation with specified parameters
    
    Parameters:
    - rainfall_mm: Rainfall amount in mm
    - duration_hours: Simulation duration in hours
    - initial_level: Initial water level percentage
    
    Returns:
    - Simulation results over time
    """
    # Reset to initial level
    simulator.dam_data["current_level"] = request.initial_level
    
    results = []
    rainfall_per_hour = request.rainfall_mm / request.duration_hours
    
    for hour in range(request.duration_hours):
        result = simulator.update_water_level(rainfall_mm=rainfall_per_hour)
        results.append({
            "hour": hour + 1,
            "level": result.level_percentage,
            "storage": result.storage_volume,
            "status": result.status,
            "inflow": result.inflow_rate,
            "outflow": result.outflow_rate
        })
    
    return {
        "simulation_parameters": {
            "total_rainfall_mm": request.rainfall_mm,
            "duration_hours": request.duration_hours,
            "initial_level": request.initial_level
        },
        "results": results,
        "final_level": results[-1]["level"],
        "final_status": results[-1]["status"]
    }

@app.post("/api/rainfall/add", tags=["Rainfall"])
async def add_rainfall_data(rainfall: RainfallData):
    """
    Add rainfall data and update water level
    
    Parameters:
    - Rainfall data (timestamp, amount, location)
    
    Returns:
    - Updated water level
    """
    return simulator.update_water_level(rainfall_mm=rainfall.rainfall_mm)

@app.get("/api/status", tags=["Status"])
async def get_system_status():
    """
    Get overall system status
    
    Returns:
    - Dam status
    - Last update time
    - Alert level
    """
    dam_info = simulator.get_dam_info()
    
    return {
        "dam_name": dam_info.name,
        "current_level": dam_info.current_level,
        "status": dam_info.status,
        "last_update": simulator.last_update,
        "alert_level": "HIGH" if dam_info.current_level >= 90 else "MEDIUM" if dam_info.current_level >= 75 else "LOW"
    }

# ==================== Run Instructions ====================
if __name__ == "__main__":
    import uvicorn
    print("🌊 Starting Dam Water Level Simulation API...")
    print("📡 API will be available at: http://localhost:8000")
    print("📚 Documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)