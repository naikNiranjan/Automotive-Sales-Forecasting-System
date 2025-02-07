import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # Add this import
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted  # Add this import

# Try to import statsmodels, handle the import error if it occurs
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    st.warning("statsmodels library is not installed. Time series model will not be available.")

def clean_data_for_prophet(df):
    """Clean and prepare data for Prophet model"""
    # Create a copy to avoid modifying original data
    prophet_df = df.copy()
    
    # Ensure Date column is datetime
    prophet_df['Date'] = pd.to_datetime(prophet_df['Date'])
    
    # Remove any rows with NaN in Date or Quantity_Sold
    prophet_df = prophet_df.dropna(subset=['Date', 'Quantity_Sold'])
    
    # Aggregate by date if there are multiple entries per day
    prophet_df = prophet_df.groupby('Date')['Quantity_Sold'].sum().reset_index()
    
    # Rename columns for Prophet
    prophet_df = prophet_df.rename(columns={'Date': 'ds', 'Quantity_Sold': 'y'})
    
    # Sort by date
    prophet_df = prophet_df.sort_values('ds')
    
    return prophet_df

def create_ensemble_model(X_train, y_train):
    """Create an ensemble of multiple models"""
    # Initialize base models
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    
    # Create voting regressor
    ensemble = VotingRegressor([
        ('rf', rf_model),
        ('gb', gb_model)
    ])
    
    # Fit the ensemble
    ensemble.fit(X_train, y_train)
    
    return ensemble

def train_model(df):
    """Train multiple predictive models using the provided dataframe"""
    if 'Date' in df.columns and 'Quantity_Sold' in df.columns:
        try:
            df = df.copy()
            
            # Check if dataframe is empty or has too few samples
            if df.empty:
                st.warning("No data available for training after filtering.")
                return None, None
            
            # Ensure minimum number of samples (at least 30 days of data)
            if len(df) < 30:
                st.warning("Insufficient data for training. Need at least 30 samples.")
                return None, None
            
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Enhanced feature engineering
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Week'] = df['Date'].dt.isocalendar().week
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['DayOfMonth'] = df['Date'].dt.day
            df['Quarter'] = df['Date'].dt.quarter
            df['WeekOfYear'] = df['Date'].dt.isocalendar().week
            df['MonthEnd'] = df['Date'].dt.is_month_end.astype(int)
            df['WeekEnd'] = (df['Date'].dt.dayofweek >= 5).astype(int)
            
            # Lag features
            df['Lag1'] = df['Quantity_Sold'].shift(1)
            df['Lag7'] = df['Quantity_Sold'].shift(7)
            df['Lag30'] = df['Quantity_Sold'].shift(30)
            
            # Rolling means
            df['Rolling7'] = df['Quantity_Sold'].rolling(7).mean()
            df['Rolling30'] = df['Quantity_Sold'].rolling(30).mean()
            
            # Drop NaN values after creating features
            df = df.dropna()
            
            # Check again after feature engineering if we have enough samples
            if len(df) < 10:  # Minimum required after feature engineering
                st.warning("Insufficient data after feature engineering. Please ensure more complete data.")
                return None, None
            
            # Prepare features
            features = ['Year', 'Month', 'Week', 'DayOfWeek', 'DayOfMonth', 
                       'Quarter', 'WeekOfYear', 'MonthEnd', 'WeekEnd',
                       'Lag1', 'Lag7', 'Lag30', 'Rolling7', 'Rolling30']
            
            # Add additional features if available
            if 'Price (INR)' in df.columns:
                features.append('Price (INR)')
            if 'Current_Stock' in df.columns:
                features.append('Current_Stock')
            if 'Inventory_Turnover_Ratio' in df.columns:
                features.append('Inventory_Turnover_Ratio')
            if 'Economic_Condition' in df.columns:
                features.append('Economic_Condition')
            if 'Promotion' in df.columns:
                features.append('Promotion')
            
            X = df[features]
            y = df['Quantity_Sold']
            
            # Validate features before scaling
            if X.empty or y.empty:
                st.warning("No valid features available for training.")
                return None, None
            
            # Scale features with error handling
            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=features)
            except ValueError as e:
                st.error(f"Error in scaling features: {str(e)}")
                return None, None
            
            # Continue with model training only if we have valid scaled features
            if X_scaled is not None:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                # Create and train models with hyperparameter tuning
                rf = RandomForestRegressor(
                    n_estimators=500,  # Increased from 200
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42
                )
                
                gb = GradientBoostingRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42
                )
                
                # Create ensemble with weighted voting
                ensemble = VotingRegressor([
                    ('rf', rf),
                    ('gb', gb)
                ])
                
                # Train all models
                models = {
                    'Random Forest': rf,
                    'Gradient Boosting': gb,
                    'Ensemble': ensemble
                }
                
                model_predictions = {}
                model_metrics = {}
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    model_predictions[name] = pred
                    model_metrics[name] = {
                        'mae': mean_absolute_error(y_test, pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
                        'r2': r2_score(y_test, pred),
                        'mape': mean_absolute_percentage_error(y_test, pred)
                    }
                
                # Find best model
                best_model_name = min(model_metrics, key=lambda x: model_metrics[x]['rmse'])
                best_model = models[best_model_name]
                
                # Calculate feature importance for all models that support it
                feature_importance = {}
                for name, model in models.items():
                    if hasattr(model, 'feature_importances_'):
                        feature_importance[name] = pd.DataFrame({
                            'feature': features,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                    elif hasattr(model, 'coef_'):
                        feature_importance[name] = pd.DataFrame({
                            'feature': features,
                            'importance': abs(model.coef_)
                        }).sort_values('importance', ascending=False)
                
                # Update metrics dictionary without SHAP values
                metrics = {
                    'best_model_name': best_model_name,
                    'model_metrics': model_metrics,
                    'feature_importance': feature_importance,
                    'X_test': X_test,
                    'y_test': y_test,
                    'features': features,
                    'scaler': scaler,
                    'last_date': df['Date'].max(),
                    'models': models,
                    'last_values': {
                        'Quantity_Sold': df['Quantity_Sold'].iloc[-1],
                        'Rolling7': df['Rolling7'].iloc[-1],
                        'Rolling30': df['Rolling30'].iloc[-1]
                    }
                }
                
                return best_model, metrics
            else:
                return None, None

        except Exception as e:
            st.error(f"Error in model training: {str(e)}")
            return None, None
    else:
        st.error("Required columns missing for prediction.")
        return None, None

def train_time_series_model(df):
    """Train a time series model using the provided dataframe"""
    if 'Date' in df.columns and 'Quantity_Sold' in df.columns:
        try:
            df = df.copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            # Ensure all required columns are present
            required_columns = ['Quantity_Sold', 'Price (INR)', 'Current_Stock', 'Inventory_Turnover_Ratio', 'Economic_Condition', 'Promotion']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0  # Fill missing columns with zeros
            
            # Check if SARIMAX is available
            if 'SARIMAX' in globals():
                # Fit SARIMAX model
                exog = df[required_columns[1:]]  # Exogenous variables
                model = SARIMAX(df['Quantity_Sold'], exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                model_fit = model.fit(disp=False)
                return model_fit
            else:
                st.warning("SARIMAX model is not available. Please install statsmodels library.")
                return None

        except Exception as e:
            st.error(f"Error in time series model training: {str(e)}")
            return None
    else:
        st.error("Required columns missing for prediction.")
        return None

def make_future_predictions(model, metrics, period='1W', selected_model=None, df=None):
    """Generate future predictions for specified period and vehicle model"""
    periods_map = {
        '1W': 7,
        '2W': 14,
        '1M': 30,
        '3M': 90,
        '6M': 180,
        '1Y': 365
    }
    
    try:
        days = periods_map.get(period, 7)
        last_date = metrics['last_date']
        
        # Calculate model-specific scaling factor
        scaling_factor = 1.0
        if selected_model and selected_model != 'All Models' and df is not None:
            total_sales = df['Quantity_Sold'].sum()
            model_sales = df[df['Vehicle_Model'] == selected_model]['Quantity_Sold'].sum()
            if total_sales > 0:
                scaling_factor = model_sales / total_sales
        
        # Filter data for selected model
        if selected_model and df is not None and selected_model != 'All Models':
            df = df[df['Vehicle_Model'] == selected_model].copy()
            if df.empty:
                st.warning(f"No data available for {selected_model}")
                return None
        
        # Generate future dates
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        # Create future features DataFrame
        future_df = pd.DataFrame(index=future_dates)
        
        # Add basic time-based features that are always available
        future_df['Year'] = future_df.index.year
        future_df['Month'] = future_df.index.month
        future_df['Week'] = future_df.index.isocalendar().week
        future_df['DayOfWeek'] = future_df.index.dayofweek
        future_df['DayOfMonth'] = future_df.index.day
        future_df['Quarter'] = future_df.index.quarter
        future_df['WeekOfYear'] = future_df.index.isocalendar().week
        future_df['MonthEnd'] = future_df.index.is_month_end.astype(int)
        future_df['WeekEnd'] = (future_df.index.dayofweek >= 5).astype(int)
        
        # Add lag features using last known values
        if 'last_values' in metrics:
            last_vals = metrics['last_values']
            future_df['Lag1'] = last_vals.get('Quantity_Sold', 0)
            future_df['Lag7'] = last_vals.get('Quantity_Sold', 0)
            future_df['Lag30'] = last_vals.get('Quantity_Sold', 0)
            future_df['Rolling7'] = last_vals.get('Rolling7', 0)
            future_df['Rolling30'] = last_vals.get('Rolling30', 0)
        
        # Ensure all required features are present
        required_features = metrics['features']
        missing_features = set(required_features) - set(future_df.columns)
        
        # Fill missing features with zeros or appropriate default values
        for feature in missing_features:
            if feature in ['Current_Stock', 'Inventory_Turnover_Ratio', 'Price (INR)', 'Economic_Condition', 'Promotion']:
                # Use mean values from training data if available
                future_df[feature] = metrics.get(f'mean_{feature}', 0)
            else:
                future_df[feature] = 0
        
        # Select only the required features in the correct order
        future_features = future_df[required_features]
        
        # Scale features
        future_scaled = metrics['scaler'].transform(future_features)
        future_scaled = pd.DataFrame(future_scaled, columns=required_features)  # Ensure feature names are retained
        
        # Make predictions
        predictions = model.predict(future_scaled)
        
        # Apply scaling factor to predictions if specific model is selected
        if selected_model and selected_model != 'All Models':
            predictions = predictions * scaling_factor
        
        # Calculate model-specific inventory metrics
        if selected_model and selected_model != 'All Models':
            # Use model-specific historical patterns
            historical_std = df['Quantity_Sold'].std()
            historical_mean = df['Quantity_Sold'].mean()
            
            # Calculate safety stock based on historical variability
            safety_stock = max(historical_std * 1.645, 1)  # Minimum of 1 unit
            
            # Calculate reorder point based on lead time and average daily demand
            avg_daily_demand = historical_mean
            lead_time_days = 7  # Assuming 7-day lead time
            reorder_point = (avg_daily_demand * lead_time_days) + safety_stock
            
            # Calculate min and max stock levels
            min_stock = max(avg_daily_demand * 3, 1)  # 3 days of demand, minimum 1 unit
            max_stock = max(avg_daily_demand * 14, reorder_point * 1.5)  # 14 days or 150% of reorder point
        else:
            # Use aggregate calculations for all models
            safety_stock = np.std(predictions) * 1.645
            reorder_point = predictions.mean() * 7 + safety_stock
            min_stock = predictions.mean() * 3
            max_stock = predictions.mean() * 14
        
        return pd.DataFrame({
            'Date': future_dates,
            'Predicted_Quantity': predictions,
            'Safety_Stock': safety_stock,
            'Reorder_Point': reorder_point,
            'Min_Stock': min_stock,
            'Max_Stock': max_stock
        })
    
    except Exception as e:
        st.error(f"Error in making predictions: {str(e)}")
        return None

def make_time_series_predictions(model, period='1W'):
    """Generate future predictions using the time series model"""
    periods_map = {
        '1W': 7,
        '2W': 14,
        '1M': 30,
        '3M': 90,
        '6M': 180,
        '1Y': 365
    }
    
    try:
        days = periods_map.get(period, 7)
        
        # Generate future dates
        future_dates = pd.date_range(
            start=model.data.endog.index[-1] + pd.Timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        # Make predictions
        forecast = model.get_forecast(steps=days)
        predictions = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        return pd.DataFrame({
            'Date': future_dates,
            'Predicted_Quantity': predictions,
            'Lower_Bound': conf_int.iloc[:, 0],
            'Upper_Bound': conf_int.iloc[:, 1]
        })
    
    except Exception as e:
        st.error(f"Error in making time series predictions: {str(e)}")
        return None

def display_predictive_analytics(df, model, metrics, time_series_model=None):
    """Display comprehensive predictive analytics"""
    # Update CSS for better styling
    st.markdown("""
        <style>
        .main {
            background-color: #1e1e1e;
        }
        .prediction-card {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            margin: 15px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .metric-row {
            background-color: rgba(255, 255, 255, 0.03);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
            margin-right: 5px;
            padding: 10px 20px;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent;
            padding: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("## 🔮 Predictive Analytics & Inventory Planning")
    
    if model and metrics:
        # Model selection and period selection in the same row
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            # Add model selection
            model_options = {
                'ensemble': 'Ensemble Model (Best Accuracy)',
                'rf': 'Random Forest',
                'gb': 'Gradient Boosting',
                'ts': 'Time Series Model'
            }
            selected_algorithm = st.selectbox(
                "Select Prediction Model",
                list(model_options.keys()),
                key="selected_algorithm",
                format_func=lambda x: model_options[x]
            )
        
        with col2:
            # Add period selection
            period = st.selectbox(
                "Forecast Period",
                ['1W', '2W', '1M', '3M', '6M', '1Y'],
                key="select_period",
                format_func=lambda x: {
                    '1W': '1 Week',
                    '2W': '2 Weeks',
                    '1M': '1 Month',
                    '3M': '3 Months',
                    '6M': '6 Months',
                    '1Y': '1 Year'
                }[x]
            )
        
        with col3:
            # Add confidence level selection
            confidence_level = st.selectbox(
                "Confidence Level",
                [70, 75, 80, 85, 90, 95],
                key="select_confidence",
                index=4  # Default to 90
            )

        # Vehicle model selection if available
        if 'Vehicle_Model' in df.columns:
            vehicle_models = ['All Models'] + sorted(df['Vehicle_Model'].unique().tolist())
            selected_vehicle = st.selectbox(
                "Select Vehicle Model",
                vehicle_models,
                key="select_vehicle"
            )
        else:
            selected_vehicle = 'All Models'

        # Use selected model for predictions with enhanced error handling
        if selected_algorithm == 'ts' and time_series_model:
            predictions_df = make_time_series_predictions(time_series_model, period)
        else:
            if selected_vehicle != 'All Models':
                filtered_df = df[df['Vehicle_Model'] == selected_vehicle].copy()
                if len(filtered_df) < 30:  # Check minimum required samples
                    st.warning(f"Insufficient data for {selected_vehicle}. Need at least 30 days of data.")
                    predictions_df = None
                else:
                    # Re-train model on filtered data for updated predictions
                    new_model, new_metrics = train_model(filtered_df)
                    if new_model and new_metrics:
                        current_model = (new_metrics['models'].get(selected_algorithm)
                                         if selected_algorithm in new_metrics['models']
                                         else new_model)
                        predictions_df = make_future_predictions(current_model, new_metrics, period, selected_vehicle, filtered_df)
                    else:
                        st.warning(f"Could not train model for {selected_vehicle}. Using overall model instead.")
                        predictions_df = make_future_predictions(model, metrics, period, selected_vehicle, df)
            else:
                current_model = (metrics['models'].get(selected_algorithm)
                                 if selected_algorithm in metrics['models']
                                 else model)
                predictions_df = make_future_predictions(current_model, metrics, period, selected_vehicle, df)
        
        if predictions_df is not None:
            # Display forecast and useful graphs
            st.markdown("### 📊 Sales Forecast")
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=predictions_df['Date'],
                y=predictions_df['Predicted_Quantity'],
                name='Forecast',
                line=dict(color='#2563eb', width=2),
                hovertemplate='Date: %{x}<br>Quantity: %{y:.2f}'
            ))
            
            # Add confidence intervals with better colors
            if 'Lower_Bound' in predictions_df.columns and 'Upper_Bound' in predictions_df.columns:
                fig.add_trace(go.Scatter(
                    x=predictions_df['Date'],
                    y=predictions_df['Upper_Bound'],
                    fill=None,
                    mode='lines',
                    line=dict(color='rgba(37, 99, 235, 0.2)'),
                    name='Upper Bound'
                ))
                
                fig.add_trace(go.Scatter(
                    x=predictions_df['Date'],
                    y=predictions_df['Lower_Bound'],
                    fill='tonexty',
                    mode='lines',
                    line=dict(color='rgba(37, 99, 235, 0.2)'),
                    name='Lower Bound'
                ))
            else:
                std_dev = predictions_df['Predicted_Quantity'].std()
                z_score = {70: 1.04, 75: 1.15, 80: 1.28, 85: 1.44, 90: 1.645, 95: 1.96}[confidence_level]
                margin = std_dev * z_score
                
                fig.add_trace(go.Scatter(
                    x=predictions_df['Date'],
                    y=predictions_df['Predicted_Quantity'] + margin,
                    fill=None,
                    mode='lines',
                    line=dict(color='rgba(37, 99, 235, 0.2)'),
                    name='Upper Bound'
                ))
                
                fig.add_trace(go.Scatter(
                    x=predictions_df['Date'],
                    y=predictions_df['Predicted_Quantity'] - margin,
                    fill='tonexty',
                    mode='lines',
                    line=dict(color='rgba(37, 99, 235, 0.2)'),
                    name='Lower Bound'
                ))
            
            fig.update_layout(
                title=f'Sales Forecast for {selected_vehicle}',
                xaxis_title='Date',
                yaxis_title='Quantity',
                template='plotly_dark',  # Changed to dark template
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(255,255,255,0.1)'  # Lighter grid for dark theme
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(255,255,255,0.1)'  # Lighter grid for dark theme
                ),
                margin=dict(t=50, l=50, r=30),
                font=dict(color='white')  # White text for dark theme
            )
            st.plotly_chart(fig, use_container_width=True)

            # Add prediction intervals
            predictions_df['Lower_Bound'] = predictions_df['Predicted_Quantity'] - margin
            predictions_df['Upper_Bound'] = predictions_df['Predicted_Quantity'] + margin
            
            st.markdown("### 📦 Inventory Recommendations")
            
            st.markdown("""
                **Inventory Recommendations:**
                - **Safety Stock:** The minimum buffer stock to maintain to avoid stockouts.
                - **Reorder Point:** The inventory level at which a new order should be placed to replenish stock.
                - **Maximum Stock Level:** The maximum recommended inventory level to avoid overstocking.
                - **Min Stock:** The minimum stock level to cover 3 days of average demand.
                - **Max Stock:** The maximum stock level to cover 14 days of average demand.
                
                Use these recommendations to optimize your inventory levels and ensure you have enough stock to meet demand without overstocking.
            """)
            
            # Create inventory planning metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Recommended Safety Stock",
                    f"{predictions_df['Safety_Stock'].iloc[0]:.0f} units",
                    "Minimum buffer stock"
                )
            with col2:
                st.metric(
                    "Reorder Point",
                    f"{predictions_df['Reorder_Point'].iloc[0]:.0f} units",
                    "Place new order when stock reaches this level"
                )
            with col3:
                st.metric(
                    "Maximum Stock Level",
                    f"{predictions_df['Max_Stock'].iloc[0]:.0f} units",
                    "Maximum recommended inventory"
                )
            
            # Create inventory planning chart
            fig_inv = go.Figure()
            
            # Add inventory level traces
            fig_inv.add_trace(go.Scatter(
                x=predictions_df['Date'],
                y=predictions_df['Max_Stock'],
                name='Maximum Stock',
                line=dict(color='#10B981', width=1, dash='dash')
            ))
            
            fig_inv.add_trace(go.Scatter(
                x=predictions_df['Date'],
                y=predictions_df['Reorder_Point'],
                name='Reorder Point',
                line=dict(color='#F59E0B', width=1, dash='dash')
            ))
            
            fig_inv.add_trace(go.Scatter(
                x=predictions_df['Date'],
                y=predictions_df['Safety_Stock'],
                name='Safety Stock',
                line=dict(color='#EF4444', width=1, dash='dot')
            ))
            
            fig_inv.add_trace(go.Scatter(
                x=predictions_df['Date'],
                y=predictions_df['Predicted_Quantity'],
                name='Predicted Demand',
                line=dict(color='#3B82F6', width=2)
            ))
            
            # Update layout with dark theme
            fig_inv.update_layout(
                title='Inventory Planning Levels',
                xaxis_title='Date',
                yaxis_title='Units',
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                font=dict(color='white'),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            st.plotly_chart(fig_inv, use_container_width=True)

            # Add validation for inventory metrics
            if selected_vehicle != 'All Models':
                total_stock = df['Current_Stock'].sum()
                if predictions_df['Max_Stock'].iloc[0] > total_stock:
                    st.warning("""
                        Note: Recommended stock levels are specific to this model and represent 
                        individual model requirements, not warehouse totals.
                    """)

            if len(df) > 30:
                filtered_df = (df if selected_vehicle == 'All Models' 
                             else df[df['Vehicle_Model'] == selected_vehicle])
                seasonal_fig = analyze_seasonality(filtered_df)
                # Update seasonal plot styling
                seasonal_fig.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=800,
                    showlegend=True,
                    title_text=f"Seasonal Patterns Analysis - {selected_vehicle}",
                    font=dict(color='white')
                )
            st.plotly_chart(seasonal_fig, use_container_width=True)

def display_model_performance(metrics):
    """Display model performance comparison in a separate dropdown"""
    with st.expander("📈 Model Performance Comparison", expanded=False):
        st.markdown("### 🎯 Model Performance Comparison")
        model_metrics_df = pd.DataFrame(metrics['model_metrics']).T
        
        col1, col2 = st.columns(2)
        with col1:
            # Model metrics comparison chart
            fig_metrics = go.Figure()
            for metric in ['rmse', 'mae', 'mape']:
                fig_metrics.add_trace(go.Bar(
                    name=metric.upper(),
                    x=list(metrics['model_metrics'].keys()),
                    y=[m[metric] for m in metrics['model_metrics'].values()],
                ))
            
            fig_metrics.update_layout(
                title="Model Performance Metrics",
                template='plotly_dark',
                barmode='group',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        with col2:
            # Feature importance plot for the selected model
            selected_algorithm = st.selectbox(
                "Select Model for Feature Importance",
                list(metrics['feature_importance'].keys())
            )
            if selected_algorithm in metrics['feature_importance']:
                fig_importance = px.bar(
                    metrics['feature_importance'][selected_algorithm].head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title=f'Top 10 Important Features - {selected_algorithm}'
                )
                fig_importance.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_importance, use_container_width=True)

def analyze_seasonality(df):
    """Analyze and visualize seasonal patterns in the data"""
    df_copy = df.copy()
    df_copy.set_index('Date', inplace=True)
    
    # Create subplots for different temporal patterns
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Weekly Pattern', 'Monthly Pattern', 'Yearly Pattern')
    )
    
    # Weekly pattern
    weekly_pattern = df_copy.groupby(df_copy.index.dayofweek)['Quantity_Sold'].mean()
    fig.add_trace(go.Scatter(x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                            y=weekly_pattern.values, mode='lines+markers', name='Weekly'),
                  row=1, col=1)
    
    # Monthly pattern
    monthly_pattern = df_copy.groupby(df_copy.index.month)['Quantity_Sold'].mean()
    fig.add_trace(go.Scatter(x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                            y=monthly_pattern.values, mode='lines+markers', name='Monthly'),
                  row=1, col=2)
    
    # Yearly pattern
    yearly_pattern = df_copy.groupby(df_copy.index.year)['Quantity_Sold'].mean()
    fig.add_trace(go.Scatter(x=yearly_pattern.index, y=yearly_pattern.values,
                            mode='lines+markers', name='Yearly'),
                  row=2, col=1)
    
    fig.update_layout(height=800, showlegend=True,
                     title_text="Seasonal Patterns Analysis")
    return fig
