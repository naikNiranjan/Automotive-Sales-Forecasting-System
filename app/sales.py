import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def display_sales_trends(df):
    """Display comprehensive historical sales analysis"""
    st.markdown("## 📈 Historical Sales Performance Analysis")
    
    analysis_type = st.selectbox(
        "Select Analysis View",
        ["Sales Overview", "Product Analysis", "Revenue Analysis", "Time Distribution"]
    )
    
    # Show date range info
    date_range = f"Data from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
    st.caption(date_range)
    
    if analysis_type == "Sales Overview":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sales Volume", f"{df['Quantity_Sold'].sum():,.0f} units")
        with col2:
            st.metric("Average Daily Sales", f"{df.groupby('Date')['Quantity_Sold'].sum().mean():,.1f} units")
        with col3:
            if 'Price (INR)' in df.columns:
                total_revenue = (df['Quantity_Sold'] * df['Price (INR)']).sum()
                st.metric("Total Revenue", f"₹{total_revenue:,.2f}")
        
        # Daily sales trend
        fig = go.Figure()
        daily_sales = df.groupby('Date')['Quantity_Sold'].sum()
        fig.add_trace(go.Scatter(
            x=daily_sales.index,
            y=daily_sales.values,
            mode='lines',
            name='Daily Sales',
            line=dict(color='#2E86C1')
        ))
        fig.update_layout(
            title='Daily Sales Trend',
            height=400,
            xaxis_title='Date',
            yaxis_title='Units Sold'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly sales bar chart
        monthly_sales = df.set_index('Date').resample('M')['Quantity_Sold'].sum()
        fig_monthly = go.Figure(data=[
            go.Bar(
                x=monthly_sales.index,
                y=monthly_sales.values,
                marker_color='#3498DB'
            )
        ])
        fig_monthly.update_layout(
            title='Monthly Sales Distribution',
            height=400,
            xaxis_title='Month',
            yaxis_title='Units Sold'
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    elif analysis_type == "Product Analysis":
        if 'Vehicle_Model' in df.columns:
            # Model-wise sales analysis
            model_sales = df.groupby('Vehicle_Model')['Quantity_Sold'].sum().sort_values(ascending=False)
            
            # Top selling models
            fig_top = px.bar(
                model_sales,
                title='Sales by Vehicle Model',
                labels={'value': 'Total Units Sold', 'Vehicle_Model': 'Model'},
                color_discrete_sequence=['#2E86C1']
            )
            st.plotly_chart(fig_top, use_container_width=True)
            
            # Model share pie chart
            fig_pie = px.pie(
                values=model_sales.values,
                names=model_sales.index,
                title='Sales Distribution by Model',
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    elif analysis_type == "Revenue Analysis":
        if all(col in df.columns for col in ['Price (INR)', 'Quantity_Sold']):
            df['Revenue'] = df['Price (INR)'] * df['Quantity_Sold']
            
            # Daily revenue trend
            daily_revenue = df.groupby('Date')['Revenue'].sum()
            fig_rev = go.Figure(data=go.Scatter(
                x=daily_revenue.index,
                y=daily_revenue.values,
                mode='lines',
                line=dict(color='#27AE60')
            ))
            fig_rev.update_layout(
                title='Daily Revenue Trend',
                height=400,
                xaxis_title='Date',
                yaxis_title='Revenue (INR)'
            )
            st.plotly_chart(fig_rev, use_container_width=True)
            
            if 'Vehicle_Model' in df.columns:
                # Revenue by model
                model_revenue = df.groupby('Vehicle_Model')['Revenue'].sum().sort_values(ascending=False)
                fig_model_rev = px.bar(
                    model_revenue,
                    title='Revenue by Model',
                    labels={'value': 'Total Revenue (INR)', 'Vehicle_Model': 'Model'},
                    color_discrete_sequence=['#27AE60']
                )
                st.plotly_chart(fig_model_rev, use_container_width=True)

    elif analysis_type == "Time Distribution":
        # Sales by day of week
        weekday_sales = df.groupby(df['Date'].dt.day_name())['Quantity_Sold'].mean()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_sales = weekday_sales.reindex(weekday_order)
        
        fig_weekday = px.bar(
            weekday_sales,
            title='Average Sales by Day of Week',
            labels={'value': 'Average Units Sold', 'Date': 'Day'},
            color_discrete_sequence=['#8E44AD']
        )
        st.plotly_chart(fig_weekday, use_container_width=True)
        
        # Monthly heatmap - Fixed version
        monthly_qty = df.pivot_table(
            values='Quantity_Sold',
            index=df['Date'].dt.year,
            columns=df['Date'].dt.month,
            aggfunc='sum',
            fill_value=0
        ).reindex(columns=range(1, 13))  # Ensure all months are present
        
        # Create month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=monthly_qty.values,
            x=month_labels,
            y=monthly_qty.index,
            colorscale='YlOrRd'
        ))
        
        fig_heat.update_layout(
            title='Monthly Sales Heatmap',
            xaxis_title='Month',
            yaxis_title='Year',
            height=400
        )
        st.plotly_chart(fig_heat, use_container_width=True)
