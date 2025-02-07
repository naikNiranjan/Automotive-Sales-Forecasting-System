import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

def display_metric_card(title, value, delta=None, suffix="", prefix=""):
    """Display a metric with custom styling"""
    st.metric(
        title,
        f"{prefix}{value:,.2f}{suffix}",
        delta if delta is not None else None
    )

def create_kpi_section(df):
    """Create comprehensive KPI section with enhanced visualizations"""
    st.markdown("""
        <style>
        .big-font {
            font-size: 24px !important;
        }
        .metric-container {
            margin-bottom: 2rem;
        }
        .chart-container {
            margin-top: 1rem;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📊 Key Performance Indicators", help="Overview of key business metrics")
    
    # Main KPI Row - Reordered metrics
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    # Column 1: Sales Volume Metrics
    with col1:
        if 'Quantity_Sold' in df.columns:
            total_units = df['Quantity_Sold'].sum()
            monthly_units = df.set_index('Date').resample('M')['Quantity_Sold'].sum()
            
            if len(monthly_units) >= 2:
                units_growth = ((monthly_units.iloc[-1] / monthly_units.iloc[-2]) - 1) * 100
                units_trend = "📈" if units_growth > 0 else "📉"
            else:
                units_growth = 0
                units_trend = ""
            
            st.metric(
                "Total Units Sold",
                f"{total_units:,.0f}",
                f"{units_growth:+.1f}% {units_trend}"
            )
    
    # Column 2: Inventory Turnover
    with col2:
        if 'Inventory_Turnover_Ratio' in df.columns:
            avg_turnover = df['Inventory_Turnover_Ratio'].mean()
            if 'Date' in df.columns:
                monthly_turnover = df.set_index('Date').resample('M')['Inventory_Turnover_Ratio'].mean()
                if len(monthly_turnover) >= 2:
                    turnover_change = ((monthly_turnover.iloc[-1] / monthly_turnover.iloc[-2]) - 1) * 100
                    turnover_trend = "📈" if turnover_change > 0 else "📉"
                else:
                    turnover_change = 0
                    turnover_trend = ""
            else:
                turnover_change = 0
                turnover_trend = ""
            
            st.metric(
                "Inventory Turnover",
                f"{avg_turnover:.2f}x",
                f"{turnover_change:+.1f}% {turnover_trend}"
            )
    
    # Column 3: Current Stock
    with col3:
        if 'Current_Stock' in df.columns:
            current_stock = df['Current_Stock'].sum()
            st.metric(
                "Current Stock",
                f"{current_stock:,.0f} units"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Visual Insights Section - Updated layout
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        if 'Quantity_Sold' in df.columns:
            monthly_sales = df.set_index('Date').resample('M')['Quantity_Sold'].sum()
            fig_sales = px.bar(
                monthly_sales,
                title='Monthly Sales Trend',
                height=400  # Increased height
            )
            fig_sales.update_layout(
                margin=dict(t=30, b=30, l=30, r=30),
                showlegend=True
            )
            st.plotly_chart(fig_sales, use_container_width=True)
    
    with viz_col2:
        if 'Inventory_Turnover_Ratio' in df.columns:
            monthly_turnover = df.set_index('Date').resample('M')['Inventory_Turnover_Ratio'].mean()
            fig_turnover = px.line(
                monthly_turnover,
                title='Monthly Inventory Turnover',
                height=400  # Increased height
            )
            fig_turnover.update_layout(
                margin=dict(t=30, b=30, l=30, r=30),
                showlegend=True
            )
            st.plotly_chart(fig_turnover, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed Analysis in expander
    with st.expander("🔍 Detailed Analysis", expanded=False):
        try:
            det_col1, det_col2 = st.columns(2)
            
            with det_col1:
                if 'Vehicle_Model' in df.columns and 'Quantity_Sold' in df.columns:
                    # Top Models Analysis
                    model_performance = df.groupby('Vehicle_Model')['Quantity_Sold'].sum().sort_values(ascending=False)
                    
                    # Create and display top models chart
                    fig_top = px.bar(
                        model_performance.head(),
                        title='Top Selling Models',
                        labels={'value': 'Units Sold', 'Vehicle_Model': 'Model'}
                    )
                    st.plotly_chart(fig_top, use_container_width=True)
            
            with det_col2:
                # Stock Status Distribution
                if 'Current_Stock' in df.columns:
                    stock_status = pd.cut(
                        df['Current_Stock'],
                        bins=[-float('inf'), 10, 50, 100, float('inf')],
                        labels=['Critical', 'Low', 'Medium', 'High']
                    ).value_counts()
                    
                    fig_status = px.pie(
                        values=stock_status.values,
                        names=stock_status.index,
                        title='Stock Level Distribution',
                        color_discrete_sequence=['red', 'orange', 'yellow', 'green']
                    )
                    st.plotly_chart(fig_status, use_container_width=True)
        
        except Exception as e:
            st.warning("Some detailed analyses are not available due to missing data")
    
    # Performance Summary
    st.markdown("### 📈 Performance Summary")
    summary_cols = st.columns(3)
    
    with summary_cols[0]:
        if 'Quantity_Sold' in df.columns:
            daily_avg = df.groupby('Date')['Quantity_Sold'].sum().mean()
            st.metric("Daily Average Sales", f"{daily_avg:.1f} units")
    
    with summary_cols[1]:
        if 'Current_Stock' in df.columns and 'Quantity_Sold' in df.columns:
            try:
                stock_coverage = current_stock / (df['Quantity_Sold'].mean() * 30)
                st.metric("Stock Coverage", f"{stock_coverage:.1f} months")
            except:
                st.metric("Stock Coverage", "N/A")
    
    with summary_cols[2]:
        if 'Inventory_Turnover_Ratio' in df.columns:
            efficiency = (avg_turnover / df['Inventory_Turnover_Ratio'].median() - 1) * 100
            st.metric("Inventory Efficiency", f"{efficiency:+.1f}%")

def create_location_analysis(df):
    """Create location-based analysis if location data is available"""
    if 'Location' in df.columns:
        st.markdown("## 📍 Location Analysis")
        
        # Location Performance
        loc_col1, loc_col2 = st.columns(2)
        
        with loc_col1:
            # Sales by Location
            location_perf = df.groupby('Location').agg({
                'Quantity_Sold': 'sum',
                'Price (INR)': lambda x: (x * df.loc[x.index, 'Quantity_Sold']).sum()
            })
            
            fig_loc_sales = px.bar(
                location_perf,
                y='Quantity_Sold',
                title='Sales Volume by Location'
            )
            st.plotly_chart(fig_loc_sales, use_container_width=True)
        
        with loc_col2:
            # Revenue by Location
            fig_loc_rev = px.bar(
                location_perf,
                y='Price (INR)',
                title='Revenue by Location'
            )
            st.plotly_chart(fig_loc_rev, use_container_width=True)

def add_filters_and_date_range(df):
    """Add date range and other filters"""
    try:
        st.sidebar.markdown("### 📅 Filters")
        
        filtered_df = df.copy()
        
        # Date range filter
        if 'Date' in filtered_df.columns:
            min_date = filtered_df['Date'].min()
            max_date = filtered_df['Date'].max()
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[(filtered_df['Date'].dt.date >= start_date) & (filtered_df['Date'].dt.date <= end_date)]
        
        # Vehicle model filter if available
        if 'Vehicle_Model' in filtered_df.columns:
            models = ['All'] + list(filtered_df['Vehicle_Model'].unique())
            selected_model = st.sidebar.selectbox("Select Vehicle Model", models)
            if selected_model != 'All':
                filtered_df = filtered_df[filtered_df['Vehicle_Model'] == selected_model]
        
        return filtered_df
    except Exception as e:
        st.error(f"Error applying filters: {str(e)}")
        return df

def display_sales_analysis(df, file_label):
    """Enhanced sales analysis with interactive features"""
    try:
        st.markdown(f"### 📈 Sales Analysis for **{file_label}**")
        
        # Apply filters
        df_filtered = add_filters_and_date_range(df)
        
        # Display KPIs
        create_kpi_section(df_filtered)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Sales Trends", "Price Analysis", "Vehicle Performance"])
        
        with tab1:
            if all(col in df_filtered.columns for col in ['Date', 'Quantity_Sold']):
                # Daily sales trend
                fig_sales = px.line(df_filtered, x='Date', y='Quantity_Sold',
                                  title='Daily Sales Trend')
                fig_sales.update_layout(height=400)
                st.plotly_chart(fig_sales, use_container_width=True)
                
                # Monthly aggregation
                monthly_sales = df_filtered.set_index('Date').resample('M')['Quantity_Sold'].sum()
                fig_monthly = px.bar(monthly_sales, title='Monthly Sales Distribution')
                fig_monthly.update_layout(height=300)
                st.plotly_chart(fig_monthly, use_container_width=True)
                
                # Add pie chart for sales distribution
                sales_distribution = df_filtered.groupby('Vehicle_Model')['Quantity_Sold'].sum()
                fig_pie = px.pie(sales_distribution, values='Quantity_Sold', names=sales_distribution.index,
                                 title='Sales Distribution by Vehicle Model')
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with tab2:
            if 'Price (INR)' in df_filtered.columns:
                # Price trends
                fig_price = px.line(df_filtered, x='Date', y='Price (INR)',
                                  color='Vehicle_Model' if 'Vehicle_Model' in df_filtered.columns else None,
                                  title='Price Trends Over Time')
                st.plotly_chart(fig_price, use_container_width=True)
                
                # Price distribution
                fig_price_dist = px.histogram(df_filtered, x='Price (INR)',
                                            title='Price Distribution')
                st.plotly_chart(fig_price_dist, use_container_width=True)
        
        with tab3:
            if 'Vehicle_Model' in df_filtered.columns:
                # Model-wise performance
                model_perf = df_filtered.groupby('Vehicle_Model')['Quantity_Sold'].sum().sort_values(ascending=True)
                fig_models = px.bar(model_perf, orientation='h',
                                  title='Sales by Vehicle Model')
                st.plotly_chart(fig_models, use_container_width=True)
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"sales_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
        
        # Alerts and insights
        if 'Current_Stock' in df_filtered.columns:
            low_stock = df_filtered[df_filtered['Current_Stock'] < 10]['Vehicle_Model'].unique()
            if len(low_stock) > 0:
                st.warning(f"⚠️ Low stock alert for models: {', '.join(low_stock)}")
        
        # AI Insights
        with st.expander("🤖 AI-Powered Insights"):
            st.write("### Key Insights:")
            if 'Quantity_Sold' in df_filtered.columns:
                best_month = monthly_sales.idxmax()
                st.info(f"📌 Best performing month: {best_month.strftime('%B %Y')}")
            if 'Vehicle_Model' in df_filtered.columns:
                top_model = model_perf.index[-1]
                st.info(f"📌 Best performing model: {top_model}")
    except Exception as e:
        st.error(f"Error in sales analysis: {str(e)}")
        st.dataframe(df.head())

def calculate_inventory_metrics(df):
    """Calculate inventory metrics with aggregations"""
    metrics = {}
    
    try:
        if 'Current_Stock' in df.columns:
            metrics['total_stock'] = df['Current_Stock'].sum()
            metrics['avg_stock'] = df['Current_Stock'].mean()
            
            # Calculate stock by model if available
            if 'Vehicle_Model' in df.columns:
                metrics['stock_by_model'] = df.groupby('Vehicle_Model')['Current_Stock'].agg({
                    'current': 'sum',
                    'average': 'mean',
                    'min': 'min',
                    'max': 'max'
                }).round(2)
        
        if 'Inventory_Turnover_Ratio' in df.columns:
            metrics['avg_turnover'] = df['Inventory_Turnover_Ratio'].mean()
            
        return metrics
    except Exception as e:
        st.error(f"Error calculating inventory metrics: {str(e)}")
        return {}

def display_inventory_analysis(df, file_label):
    """Enhanced inventory analysis with detailed metrics"""
    try:
        # Apply filters first
        df_filtered = add_filters_and_date_range(df)
        
        st.markdown(f"### 📦 Inventory Analysis for **{file_label}**")
        
        # Calculate and display inventory metrics
        metrics = calculate_inventory_metrics(df_filtered)
        
        # Display KPIs in columns
        if metrics:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'total_stock' in metrics:
                    st.metric("Total Current Stock", f"{metrics['total_stock']:,.0f}")
            
            with col2:
                if 'avg_stock' in metrics:
                    st.metric("Average Stock Level", f"{metrics['avg_stock']:.1f}")
            
            with col3:
                if 'avg_turnover' in metrics:
                    st.metric("Avg Turnover Ratio", f"{metrics['avg_turnover']:.2f}")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Stock Trends", "Model-wise Analysis"])
        
        with tab1:
            if "Date" in df_filtered.columns and "Current_Stock" in df_filtered.columns:
                # Daily stock levels
                fig_stock = px.line(df_filtered, 
                                  x="Date", 
                                  y="Current_Stock",
                                  title="Stock Levels Over Time")
                fig_stock.update_layout(height=400)
                st.plotly_chart(fig_stock, use_container_width=True)
                
                # Monthly average stock
                monthly_stock = df_filtered.groupby(
                    pd.Grouper(key='Date', freq='M'))['Current_Stock'].mean()
                fig_monthly = px.bar(monthly_stock, 
                                   title='Monthly Average Stock Levels')
                st.plotly_chart(fig_monthly, use_container_width=True)
        
        with tab2:
            if 'stock_by_model' in metrics:
                st.markdown("### Stock Levels by Model")
                st.dataframe(metrics['stock_by_model'], use_container_width=True)
                
                # Visualize current stock by model
                fig_models = px.bar(
                    metrics['stock_by_model'].reset_index(),
                    x='Vehicle_Model',
                    y='current',
                    title='Current Stock by Model'
                )
                st.plotly_chart(fig_models, use_container_width=True)
        
        # Stock alerts
        if 'Current_Stock' in df_filtered.columns:
            low_stock_threshold = st.slider("Low Stock Alert Threshold", 
                                         min_value=1, 
                                         max_value=50, 
                                         value=10)
            
            low_stock = df_filtered[
                df_filtered['Current_Stock'] < low_stock_threshold
            ]['Vehicle_Model'].unique()
            
            if len(low_stock) > 0:
                st.warning(f"⚠️ Low stock alert for models: {', '.join(low_stock)}")
                
                # Show detailed low stock info
                low_stock_df = df_filtered[
                    df_filtered['Vehicle_Model'].isin(low_stock)
                ][['Vehicle_Model', 'Current_Stock']].drop_duplicates()
                st.dataframe(low_stock_df)
        
        # Download option
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Inventory Report",
            data=csv,
            file_name=f"inventory_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
        
    except Exception as e:
        st.error(f"Error in inventory analysis: {str(e)}")
        st.write("Debug info:")
        st.write("Columns available:", df.columns.tolist())
        st.dataframe(df.head())

def display_promotional_analysis(df, file_label):
    st.markdown(f"### Promotional Data Analysis for **{file_label}**")
    if "Discount_Percentage" in df.columns:
        st.write("Average Discount Percentage:", np.round(df["Discount_Percentage"].mean(), 2))
    if "Promotion_Type" in df.columns:
        promo_counts = df["Promotion_Type"].value_counts().reset_index()
        promo_counts.columns = ["Promotion_Type", "Count"]
        fig_promo = px.bar(promo_counts, x="Promotion_Type", y="Count", title="Promotion Type Counts")
        st.plotly_chart(fig_promo, use_container_width=True)
    if "Promotion_Start_Date" in df.columns:
        if "Date" not in df.columns:
            df["Date"] = pd.to_datetime(df["Promotion_Start_Date"], dayfirst=True, errors="coerce")
        fig_promo_time = px.line(df, x="Date", y="Discount_Percentage" if "Discount_Percentage" in df.columns else None,
                                 title="Promotional Trends Over Time")
        st.plotly_chart(fig_promo_time, use_container_width=True)
    st.dataframe(df.head())

def display_external_analysis(df, file_label):
    st.markdown(f"### External Factors Analysis for **{file_label}**")
    if "Date" in df.columns and "Regional_Demand_Index" in df.columns:
        fig_demand = px.line(df, x="Date", y="Regional_Demand_Index", title="Regional Demand Index Over Time")
        st.plotly_chart(fig_demand, use_container_width=True)
    st.dataframe(df.head())

def display_unknown_analysis(df, file_label):
    st.markdown(f"### Data Preview for **{file_label}** (Unknown Type)")
    st.dataframe(df.head())
