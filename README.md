# Dynamic Pricing for Urban Parking Lots

## 🚗 Project Overview

This project implements an intelligent dynamic pricing system for urban parking lots as part of the Summer Analytics 2025 Capstone Project. The system uses real-time data to optimize parking prices based on demand, competition, and various environmental factors.

## 📋 Problem Statement

Urban parking spaces are limited resources that need dynamic pricing to optimize utilization. Static pricing leads to either overcrowding or underutilization. This project creates a data-driven pricing engine using real-time streams and machine learning models.

## 🎯 Objectives

- Build three pricing models of increasing complexity
- Implement real-time data processing with Pathway
- Create interactive visualizations with Bokeh
- Provide competitive intelligence and routing suggestions
- Ensure smooth, explainable, and bounded price variations

## 📊 Dataset

The dataset contains 18,368 records from 14 urban parking spaces over 73 days with:
- **Location**: Latitude, Longitude
- **Capacity**: Maximum vehicles
- **Occupancy**: Current vehicles parked
- **Queue Length**: Vehicles waiting
- **Vehicle Type**: Car, bike, truck, cycle
- **Traffic Conditions**: Low, average, high
- **Special Days**: Holidays/events indicator
- **Timestamps**: Date and time information

## 🧠 Models Implemented

### Model 1: Baseline Linear Pricing
```python
Price = Previous_Price + α × (Occupancy / Capacity)
```
- Simple linear relationship with occupancy rate
- Acts as baseline for comparison
- Parameters: α (learning rate), min/max price bounds

### Model 2: Demand-Based Pricing Function
```python
Demand = α×Occupancy_Rate + β×Queue + γ×Traffic + δ×Special_Day + ε×Vehicle_Weight
Price = Base_Price × (1 + λ×Normalized_Demand) × Vehicle_Weight
```
- Multi-factor demand calculation
- Vehicle-specific pricing weights
- Traffic condition multipliers
- Special day premiums
- Bounded using tanh normalization

### Model 3: Competitive Pricing (Optional)
- Geographic proximity analysis using Haversine formula
- Competitor price comparison
- Routing suggestions when lots are full
- Dynamic competitive adjustments

## 🛠️ Implementation Features

### Real-time Processing
- **Pathway Integration**: Streaming data pipeline
- **Live Updates**: Continuous price recalculation
- **Scalable Architecture**: Handle multiple parking lots

### Interactive Visualizations
- **Bokeh Dashboard**: Real-time price monitoring
- **Comparative Analysis**: Model performance comparison
- **Demand Tracking**: Visual demand score evolution
- **Geographic Mapping**: Competitive landscape view

### Business Intelligence
- **Routing Suggestions**: Direct customers to available lots
- **Competitive Analysis**: Monitor nearby pricing
- **Demand Prediction**: Identify peak usage patterns
- **Revenue Optimization**: Maximize lot utilization

## 📁 File Structure

```
├── Complete_Dynamic_Pricing_Implementation.ipynb  # Main implementation notebook
├── dataset.csv                                   # Parking data
├── problem statement.pdf                         # Original requirements
├── parking_pricing_analysis.png                  # Analysis visualizations
├── model_comparison.csv                          # Results output
|── pathway_streaming_result.csv
|── comprehensive_pricing_results.csv
└── README.md                                     # This file
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib bokeh pathway scikit-learn
```

### Running the Implementation

1. **Local Environment**:
   ```python
   # Run the standalone implementation
   python dynamic_pricing_implementation.py
   ```

2. **Jupyter Notebook**:
   - Open `Complete_Dynamic_Pricing_Implementation.ipynb`
   - Run all cells sequentially
   - Interactive visualizations will appear inline

3. **Google Colab**:
   - Upload the notebook to Colab
   - Mount Google Drive and update data path
   - Install packages and run

### Quick Start Example
```python
from dynamic_pricing_implementation import DynamicParkingPricer

# Initialize pricer
pricer = DynamicParkingPricer(base_price=10.0, min_price=5.0, max_price=20.0)

# Model 1: Linear pricing
price_linear = pricer.model1_linear_pricing(
    prev_price=10.0, occupancy=300, capacity=500
)

# Model 2: Demand-based pricing  
price_demand, demand_score = pricer.model2_demand_based_pricing(
    occupancy=300, capacity=500, queue_length=5,
    traffic_condition='high', is_special_day=1, vehicle_type='car'
)

print(f"Linear Model Price: ${price_linear:.2f}")
print(f"Demand Model Price: ${price_demand:.2f}")
```

## 📈 Results Summary

### Model Performance
| Model | Avg Price | Price Range | Std Dev |
|-------|-----------|-------------|---------|
| Linear | $14.10 | $6.01-$20.00 | $4.23 |
| Demand | $11.71 | $5.23-$20.00 | $3.45 |
| Competitive | $11.47 | $5.12-$19.60 | $3.38 |

### Key Insights
1. **Vehicle Type Impact**: Trucks pay 50% more than cars
2. **Traffic Effect**: High traffic increases prices by 20%
3. **Demand Sensitivity**: Model 2 provides more nuanced pricing
4. **Competitive Intelligence**: Model 3 optimizes for market position

## 🔧 Configuration

### Pricing Parameters
```python
# Model 1 Parameters
ALPHA = 5.0  # Occupancy sensitivity

# Model 2 Parameters  
BASE_PRICE = 10.0
MIN_PRICE = 5.0
MAX_PRICE = 20.0
VEHICLE_WEIGHTS = {'car': 1.0, 'truck': 1.5, 'bike': 0.7, 'cycle': 0.5}
TRAFFIC_MULTIPLIERS = {'low': 0.9, 'average': 1.0, 'high': 1.2}

# Model 3 Parameters
PROXIMITY_THRESHOLD = 2.0  # km radius for competitors
```

## 📊 Visualizations

The implementation includes several visualization types:

1. **Real-time Price Tracking**: Monitor prices across multiple lots
2. **Demand Analysis**: Visualize demand scores over time  
3. **Occupancy Correlation**: Price vs occupancy rate relationships
4. **Vehicle Type Comparison**: Pricing differences by vehicle
5. **Traffic Impact**: How traffic affects pricing decisions

## 🌊 Pathway Integration

For real-time streaming:

```python
# Create data stream
stream = pw.debug.table_from_pandas(parking_data)

# Apply pricing transformation
priced_stream = stream.select(
    price=pw.apply(calculate_dynamic_price, 
                   stream.occupancy, stream.capacity, stream.queue_length)
)

# Output to dashboard
pw.io.csv.write(priced_stream, "live_prices.csv")
```

## 🔍 Testing

Run the test scenarios:
```python
# Test different occupancy levels
test_scenarios = [
    (100, 500),  # 20% occupancy
    (300, 500),  # 60% occupancy  
    (450, 500),  # 90% occupancy
    (500, 500),  # 100% occupancy
]

for occupancy, capacity in test_scenarios:
    price = pricer.model2_demand_based_pricing(
        occupancy, capacity, 5, 'average', 0, 'car'
    )
    print(f"Occupancy: {occupancy/capacity:.0%} → Price: ${price[0]:.2f}")
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create Pull Request

## 📝 License

This project is part of the Summer Analytics 2025 program. Please refer to the program guidelines for usage terms.

## 🙏 Acknowledgments

- **Consulting & Analytics Club**: Project hosting and support
- **Pathway**: Real-time data processing framework
- **Bokeh**: Interactive visualization capabilities
- **Summer Analytics 2025**: Educational program platform

## 📞 Support

For questions or issues:
1. Check the notebook documentation
2. Review the example implementations
3. Consult the problem statement PDF
4. Reach out to the Summer Analytics team

---

## 🎯 Quick Links

- [📓 Complete Implementation Notebook](Complete_Dynamic_Pricing_Implementation.ipynb)
- [📊 Analysis Results](comprehensive_pricing_analysis.png)
- [📄 Problem Statement](problem%20statement.pdf)

**Ready to revolutionize urban parking with intelligent pricing! 🚀**