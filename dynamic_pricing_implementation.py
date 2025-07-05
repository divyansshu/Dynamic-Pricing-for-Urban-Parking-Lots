#!/usr/bin/env python3
"""
Dynamic Pricing for Urban Parking Lots - Implementation
Complete implementation according to the project requirements
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pathway as pw
from bokeh.plotting import figure, show
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource
from bokeh.io import curdoc, push_notebook
import warnings
warnings.filterwarnings('ignore')

class DynamicParkingPricer:
    """Dynamic pricing system for urban parking lots"""
    
    def __init__(self, base_price=10.0, min_price=5.0, max_price=20.0):
        self.base_price = base_price
        self.min_price = min_price
        self.max_price = max_price
        
        # Vehicle type weights for pricing
        self.vehicle_weights = {
            'car': 1.0,
            'truck': 1.5,  # Trucks pay more due to space usage
            'bike': 0.7,   # Bikes pay less
            'cycle': 0.5   # Cycles pay least
        }
        
        # Traffic condition multipliers
        self.traffic_multipliers = {
            'low': 0.9,
            'average': 1.0,
            'high': 1.2
        }
        
        # Store historical prices for each parking lot
        self.price_history = {}
    
    def model1_linear_pricing(self, prev_price, occupancy, capacity, alpha=5.0):
        """
        Model 1: Baseline Linear Pricing Model
        Price = Previous_Price + α * (Occupancy / Capacity)
        """
        if capacity == 0:
            occupancy_rate = 0
        else:
            occupancy_rate = occupancy / capacity
        
        price_increment = alpha * occupancy_rate
        new_price = prev_price + price_increment
        
        # Ensure price stays within bounds
        new_price = np.clip(new_price, self.min_price, self.max_price)
        return new_price
    
    def model2_demand_based_pricing(self, occupancy, capacity, queue_length, 
                                   traffic_condition, is_special_day, vehicle_type):
        """
        Model 2: Demand-Based Pricing Function
        More sophisticated model considering multiple demand factors
        """
        # Calculate occupancy rate
        occupancy_rate = occupancy / capacity if capacity > 0 else 0
        
        # Normalize queue length (assuming max reasonable queue is 20)
        normalized_queue = min(queue_length / 20.0, 1.0)
        
        # Get vehicle weight
        vehicle_weight = self.vehicle_weights.get(vehicle_type, 1.0)
        
        # Get traffic multiplier
        traffic_multiplier = self.traffic_multipliers.get(traffic_condition, 1.0)
        
        # Special day premium
        special_day_multiplier = 1.3 if is_special_day else 1.0
        
        # Demand function components
        alpha = 8.0    # Occupancy impact
        beta = 3.0     # Queue impact
        gamma = 2.0    # Traffic impact
        delta = 1.0    # Special day impact
        epsilon = 1.0  # Vehicle type impact
        
        # Calculate demand score
        demand_score = (
            alpha * occupancy_rate +
            beta * normalized_queue +
            gamma * (traffic_multiplier - 1.0) +
            delta * (special_day_multiplier - 1.0) +
            epsilon * (vehicle_weight - 1.0)
        )
        
        # Normalize demand to prevent extreme pricing
        normalized_demand = np.tanh(demand_score / 10.0)  # Scale and bound between -1 and 1
        
        # Calculate price based on demand
        price = self.base_price * (1 + 0.5 * normalized_demand) * vehicle_weight
        
        # Apply bounds
        price = np.clip(price, self.min_price, self.max_price)
        
        return price, demand_score
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def model3_competitive_pricing(self, current_lot_data, all_lots_data, 
                                  current_lot_price, proximity_threshold=2.0):
        """
        Model 3: Competitive Pricing Model (Optional)
        Considers nearby competitor prices and suggests routing
        """
        current_lat = current_lot_data['Latitude']
        current_lon = current_lot_data['Longitude']
        current_occupancy_rate = current_lot_data['Occupancy'] / current_lot_data['Capacity']
        
        # Find nearby competitors
        nearby_lots = []
        for _, lot in all_lots_data.iterrows():
            if lot['SystemCodeNumber'] != current_lot_data['SystemCodeNumber']:
                distance = self.calculate_distance(
                    current_lat, current_lon, 
                    lot['Latitude'], lot['Longitude']
                )
                if distance <= proximity_threshold:
                    nearby_lots.append({
                        'system': lot['SystemCodeNumber'],
                        'distance': distance,
                        'occupancy_rate': lot['Occupancy'] / lot['Capacity'],
                        'capacity': lot['Capacity'],
                        'occupancy': lot['Occupancy']
                    })
        
        if not nearby_lots:
            return current_lot_price, None  # No competitors nearby
        
        # Calculate average competitor occupancy
        avg_competitor_occupancy = np.mean([lot['occupancy_rate'] for lot in nearby_lots])
        
        # Competitive adjustment
        if current_occupancy_rate > 0.9:  # Current lot is nearly full
            if avg_competitor_occupancy < 0.7:  # Competitors have space
                # Suggest rerouting to less occupied nearby lots
                best_alternative = min(nearby_lots, key=lambda x: x['occupancy_rate'])
                suggestion = f"Consider lot {best_alternative['system']} ({best_alternative['distance']:.1f}km away, {best_alternative['occupancy_rate']:.1%} occupied)"
                # Slightly reduce price to remain competitive
                adjusted_price = current_lot_price * 0.95
            else:
                suggestion = None
                adjusted_price = current_lot_price * 1.1  # All lots busy, increase price
        else:
            # Current lot has space
            if avg_competitor_occupancy > 0.8:  # Competitors are busier
                adjusted_price = current_lot_price * 1.05  # Slight premium
            else:
                adjusted_price = current_lot_price * 0.98  # Stay competitive
            suggestion = None
        
        adjusted_price = np.clip(adjusted_price, self.min_price, self.max_price)
        return adjusted_price, suggestion

class ParkingDataStream:
    """Simulates real-time data streaming using Pathway"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.df['DateTime'] = pd.to_datetime(
            self.df['LastUpdatedDate'] + ' ' + self.df['LastUpdatedTime'],
            format='%d-%m-%Y %H:%M:%S'
        )
        self.df = self.df.sort_values('DateTime')
        
    def create_pathway_stream(self):
        """Create a Pathway table for streaming data simulation"""
        # Convert DataFrame to format suitable for Pathway streaming
        # This simulates real-time data ingestion
        
        # For demonstration, we'll use a subset of data
        stream_data = self.df.head(1000).copy()  # Use first 1000 records for demo
        
        # Create Pathway table
        # Note: In a real implementation, this would connect to a live data source
        return pw.debug.table_from_pandas(stream_data)

def create_visualizations(pricing_data):
    """Create Bokeh visualizations for real-time pricing"""
    
    # Prepare data for visualization
    source = ColumnDataSource(pricing_data)
    
    # Create price trend plot
    price_plot = figure(
        title="Real-time Parking Prices by Location",
        x_axis_label="Time",
        y_axis_label="Price ($)",
        width=800,
        height=400,
        x_axis_type='datetime'
    )
    
    # Plot prices for different parking systems
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 
             'gray', 'olive', 'cyan', 'yellow', 'magenta', 'black', 'navy']
    
    unique_systems = pricing_data['SystemCodeNumber'].unique()
    for i, system in enumerate(unique_systems[:5]):  # Show first 5 for clarity
        system_data = pricing_data[pricing_data['SystemCodeNumber'] == system]
        price_plot.line(
            system_data['DateTime'], 
            system_data['Price'], 
            legend_label=f"Lot {system}", 
            color=colors[i % len(colors)],
            line_width=2
        )
    
    # Create occupancy vs price scatter plot
    occupancy_plot = figure(
        title="Price vs Occupancy Rate",
        x_axis_label="Occupancy Rate (%)",
        y_axis_label="Price ($)",
        width=400,
        height=400
    )
    
    occupancy_plot.circle(
        pricing_data['Occupancy'] / pricing_data['Capacity'] * 100,
        pricing_data['Price'],
        size=8,
        alpha=0.6,
        color='blue'
    )
    
    # Create demand score plot
    if 'DemandScore' in pricing_data.columns:
        demand_plot = figure(
            title="Demand Score Over Time",
            x_axis_label="Time",
            y_axis_label="Demand Score",
            width=400,
            height=400,
            x_axis_type='datetime'
        )
        
        demand_plot.line(
            pricing_data['DateTime'],
            pricing_data['DemandScore'],
            color='red',
            line_width=2
        )
    else:
        demand_plot = None
    
    return price_plot, occupancy_plot, demand_plot

def process_parking_data(df_sample):
    """Process parking data and apply pricing models"""
    
    pricer = DynamicParkingPricer()
    results = []
    
    # Group by parking system for processing
    for system_code in df_sample['SystemCodeNumber'].unique():
        system_data = df_sample[df_sample['SystemCodeNumber'] == system_code].copy()
        system_data = system_data.sort_values('DateTime')
        
        prev_price = pricer.base_price  # Start with base price
        
        for idx, row in system_data.iterrows():
            # Model 1: Linear pricing
            price_m1 = pricer.model1_linear_pricing(
                prev_price, row['Occupancy'], row['Capacity']
            )
            
            # Model 2: Demand-based pricing
            price_m2, demand_score = pricer.model2_demand_based_pricing(
                row['Occupancy'], row['Capacity'], row['QueueLength'],
                row['TrafficConditionNearby'], row['IsSpecialDay'], row['VehicleType']
            )
            
            # Model 3: Competitive pricing (simplified - using Model 2 as base)
            # In a full implementation, this would consider all nearby lots
            price_m3, suggestion = pricer.model3_competitive_pricing(
                row, df_sample, price_m2
            )
            
            results.append({
                'DateTime': row['DateTime'],
                'SystemCodeNumber': row['SystemCodeNumber'],
                'Capacity': row['Capacity'],
                'Occupancy': row['Occupancy'],
                'OccupancyRate': row['Occupancy'] / row['Capacity'],
                'QueueLength': row['QueueLength'],
                'VehicleType': row['VehicleType'],
                'TrafficCondition': row['TrafficConditionNearby'],
                'IsSpecialDay': row['IsSpecialDay'],
                'Price_Model1': price_m1,
                'Price_Model2': price_m2,
                'Price_Model3': price_m3,
                'DemandScore': demand_score,
                'Price': price_m2,  # Use Model 2 as primary price
                'Suggestion': suggestion
            })
            
            prev_price = price_m2  # Update previous price for next iteration
    
    return pd.DataFrame(results)

def main():
    """Main function to run the dynamic pricing system"""
    
    print("🚗 Dynamic Pricing for Urban Parking Lots")
    print("=" * 50)
    
    # Load data
    print("📊 Loading parking data...")
    df = pd.read_csv('dataset.csv')
    
    # Add datetime column
    df['DateTime'] = pd.to_datetime(
        df['LastUpdatedDate'] + ' ' + df['LastUpdatedTime'],
        format='%d-%m-%Y %H:%M:%S'
    )
    
    print(f"✅ Loaded {len(df)} records from {df['SystemCodeNumber'].nunique()} parking systems")
    
    # Use a sample for demonstration (first day of data)
    first_day = df['LastUpdatedDate'].min()
    df_sample = df[df['LastUpdatedDate'] == first_day].copy()
    df_sample = df_sample.sort_values('DateTime')
    
    print(f"📅 Processing sample data for {first_day}: {len(df_sample)} records")
    
    # Process data with pricing models
    print("💰 Applying dynamic pricing models...")
    pricing_results = process_parking_data(df_sample)
    
    print("\n🏆 Pricing Model Results Summary:")
    print(f"Model 1 (Linear) - Average Price: ${pricing_results['Price_Model1'].mean():.2f}")
    print(f"Model 2 (Demand)  - Average Price: ${pricing_results['Price_Model2'].mean():.2f}")
    print(f"Model 3 (Competitive) - Average Price: ${pricing_results['Price_Model3'].mean():.2f}")
    
    # Show price ranges
    print(f"\nPrice Ranges:")
    print(f"Model 1: ${pricing_results['Price_Model1'].min():.2f} - ${pricing_results['Price_Model1'].max():.2f}")
    print(f"Model 2: ${pricing_results['Price_Model2'].min():.2f} - ${pricing_results['Price_Model2'].max():.2f}")
    print(f"Model 3: ${pricing_results['Price_Model3'].min():.2f} - ${pricing_results['Price_Model3'].max():.2f}")
    
    # Basic matplotlib visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Price comparison across models
    plt.subplot(2, 3, 1)
    plt.plot(pricing_results['DateTime'], pricing_results['Price_Model1'], 
             label='Model 1 (Linear)', alpha=0.7)
    plt.plot(pricing_results['DateTime'], pricing_results['Price_Model2'], 
             label='Model 2 (Demand)', alpha=0.7)
    plt.plot(pricing_results['DateTime'], pricing_results['Price_Model3'], 
             label='Model 3 (Competitive)', alpha=0.7)
    plt.title('Price Comparison Across Models')
    plt.xlabel('Time')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot 2: Occupancy rate distribution
    plt.subplot(2, 3, 2)
    plt.hist(pricing_results['OccupancyRate'], bins=20, alpha=0.7, color='skyblue')
    plt.title('Occupancy Rate Distribution')
    plt.xlabel('Occupancy Rate')
    plt.ylabel('Frequency')
    
    # Plot 3: Price vs Occupancy
    plt.subplot(2, 3, 3)
    plt.scatter(pricing_results['OccupancyRate'], pricing_results['Price'], alpha=0.6)
    plt.title('Price vs Occupancy Rate')
    plt.xlabel('Occupancy Rate')
    plt.ylabel('Price ($)')
    
    # Plot 4: Demand score over time
    plt.subplot(2, 3, 4)
    plt.plot(pricing_results['DateTime'], pricing_results['DemandScore'], color='red')
    plt.title('Demand Score Over Time')
    plt.xlabel('Time')
    plt.ylabel('Demand Score')
    plt.xticks(rotation=45)
    
    # Plot 5: Queue length impact
    plt.subplot(2, 3, 5)
    plt.scatter(pricing_results['QueueLength'], pricing_results['Price'], alpha=0.6, color='orange')
    plt.title('Price vs Queue Length')
    plt.xlabel('Queue Length')
    plt.ylabel('Price ($)')
    
    # Plot 6: Vehicle type pricing
    plt.subplot(2, 3, 6)
    vehicle_prices = pricing_results.groupby('VehicleType')['Price'].mean()
    plt.bar(vehicle_prices.index, vehicle_prices.values, color=['blue', 'green', 'red', 'orange'])
    plt.title('Average Price by Vehicle Type')
    plt.xlabel('Vehicle Type')
    plt.ylabel('Average Price ($)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('parking_pricing_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Show sample suggestions from Model 3
    suggestions = pricing_results[pricing_results['Suggestion'].notna()]
    if not suggestions.empty:
        print(f"\n💡 Sample Routing Suggestions from Model 3:")
        for _, row in suggestions.head(3).iterrows():
            print(f"⏰ {row['DateTime'].strftime('%H:%M')} - {row['Suggestion']}")
    
    # Save results
    pricing_results.to_csv('pricing_results.csv', index=False)
    print(f"\n💾 Results saved to 'pricing_results.csv'")
    
    return pricing_results

if __name__ == "__main__":
    results = main()