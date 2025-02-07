import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from datetime import datetime, timedelta
import chardet
import csv
import io
import utils  # Change this line from 'from app import utils'
import kpi
import sales  # Change this line to use proper import
import predictive  # Import the new predictive module
import google.generativeai as genai  # Import the generative AI module
import chat  # Import the chat module

# Configure your Google API key (replace with your actual API key)
GOOGLE_API_KEY = "AIzaSyCxvcEQdd8gON0uigqaA7WG-N1IJrkejos"
genai.configure(api_key=GOOGLE_API_KEY)

# Must be the first Streamlit command
st.set_page_config(
    page_title="Business Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 1rem;
        background-color: #ffffff; /* Change background to white */
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Session State Initialization for Uploaded Files
# --------------------------
def init_session_state():
    """Initialize or reset session state variables"""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'file_dfs' not in st.session_state:
        st.session_state.file_dfs = {}

def remove_file(filename):
    """Remove file from session state and cached dataframes"""
    if filename in st.session_state.file_dfs:
        del st.session_state.file_dfs[filename]
    st.session_state.uploaded_files = [
        f for f in st.session_state.uploaded_files if f.name != filename
    ]
    # Trigger rerun after file removal
    st.rerun()

# Initialize session state
init_session_state()

# --------------------------
# App Title & Description
# --------------------------
st.title("Sales Forecasting & Inventory Recommendation Dashboard")

st.markdown("""
This app lets you upload multiple CSV files containing your inventory, promotional, external, or sales data.
You can remove files you don't want, and the app will dynamically analyze each file based on its contents.
If you upload only one file, the app will show analysis for that file without attempting to merge.
""")

def display_enhanced_kpi_metrics(df):
    """Enhanced KPI display with detailed metrics and trends"""
    st.markdown("## 📊 Key Performance Indicators")
    
    # Create main KPI columns
    col1, col2, col3 = st.columns(3)
    
    # Column 1: Sales Metrics
    with col1:
        if 'Price (INR)' in df.columns and 'Quantity_Sold' in df.columns:
            total_revenue = (df['Price (INR)'] * df['Quantity_Sold']).sum()
            monthly_revenue = df.set_index('Date').resample('M')[['Price (INR)', 'Quantity_Sold']].sum()
            monthly_revenue['Revenue'] = monthly_revenue['Price (INR)'] * monthly_revenue['Quantity_Sold']
            
            if len(monthly_revenue) >= 2:
                revenue_growth = ((monthly_revenue['Revenue'].iloc[-1] / monthly_revenue['Revenue'].iloc[-2]) - 1) * 100
                revenue_trend = "📈" if revenue_growth > 0 else "📉"
            else:
                revenue_growth = 0
                revenue_trend = ""
            
            st.metric(
                "Total Revenue",
                f"₹{total_revenue:,.2f}",
                f"{revenue_growth:+.1f}% {revenue_trend}"
            )
            
            # Add revenue trend chart
            fig_revenue = px.line(monthly_revenue, y='Revenue', title='Monthly Revenue Trend')
            st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Column 2: Sales Volume Metrics
    with col2:
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
            
            # Add units sold trend chart
            fig_units = px.bar(monthly_units, title='Monthly Units Sold')
            st.plotly_chart(fig_units, use_container_width=True)
    
    # Column 3: Inventory Metrics
    with col3:
        if 'Current_Stock' in df.columns:
            current_stock = df['Current_Stock'].sum()
            if 'Inventory_Turnover_Ratio' in df.columns:
                avg_turnover = df['Inventory_Turnover_Ratio'].mean()
                st.metric(
                    "Inventory Health",
                    f"{current_stock:,.0f} units",
                    f"Turnover: {avg_turnover:.2f}x 🔄"
                )
                
                # Add inventory turnover chart
                monthly_turnover = df.set_index('Date').resample('M')['Inventory_Turnover_Ratio'].mean()
                fig_turnover = px.line(monthly_turnover, title='Monthly Inventory Turnover')
                st.plotly_chart(fig_turnover, use_container_width=True)
    
    # Additional KPI Details
    with st.expander("📊 Detailed KPI Analysis"):
        # Create three columns for detailed metrics
        det_col1, det_col2, det_col3 = st.columns(3)
        
        with det_col1:
            if 'Price (INR)' in df.columns and 'Quantity_Sold' in df.columns:
                avg_order_value = total_revenue / total_units if total_units > 0 else 0
                st.metric("Average Order Value", f"₹{avg_order_value:,.2f}")
                
                # Daily revenue stats
                daily_revenue = df.groupby('Date').apply(lambda x: (x['Price (INR)'] * x['Quantity_Sold']).sum())
                st.metric("Daily Avg Revenue", f"₹{daily_revenue.mean():,.2f}")
                st.metric("Peak Daily Revenue", f"₹{daily_revenue.max():,.2f}")
        
        with det_col2:
            if 'Quantity_Sold' in df.columns:
                # Sales velocity metrics
                daily_sales = df.groupby('Date')['Quantity_Sold'].sum()
                st.metric("Daily Avg Units", f"{daily_sales.mean():.1f}")
                st.metric("Peak Daily Sales", f"{daily_sales.max():,.0f}")
                st.metric("Sales Days", f"{len(daily_sales)}")
        
        with det_col3:
            if 'Current_Stock' in df.columns:
                # Stock metrics
                if 'Vehicle_Model' in df.columns:
                    models_in_stock = df[df['Current_Stock'] > 0]['Vehicle_Model'].nunique()
                    st.metric("Models in Stock", f"{models_in_stock}")
                
                # Stock level indicators
                low_stock = df[df['Current_Stock'] < 10]['Vehicle_Model'].nunique()
                st.metric("Low Stock Models", f"{low_stock}", "Need attention! ⚠️" if low_stock > 0 else "All good ✅")
    
    # Performance Insights
    st.markdown("### 🎯 Performance Insights")
    
    # Create columns for insights
    ins_col1, ins_col2 = st.columns(2)
    
    with ins_col1:
        if 'Vehicle_Model' in df.columns and 'Quantity_Sold' in df.columns:
            # Top performing models
            top_models = df.groupby('Vehicle_Model')['Quantity_Sold'].sum().sort_values(ascending=False).head(5)
            fig_top = px.bar(top_models, title='Top 5 Selling Models')
            st.plotly_chart(fig_top, use_container_width=True)
    
    with ins_col2:
        if 'Date' in df.columns and 'Quantity_Sold' in df.columns:
            # Sales heatmap by weekday and hour
            sales_by_day = df.groupby(df['Date'].dt.day_name())['Quantity_Sold'].mean()
            fig_heatmap = px.bar(sales_by_day, title='Average Sales by Day of Week')
            st.plotly_chart(fig_heatmap, use_container_width=True)

def display_inventory_insights(df):
    st.markdown("## 📦 Inventory Management Insights")
    
    if 'Current_Stock' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Stock levels by model
            stock_by_model = df.groupby('Vehicle_Model')['Current_Stock'].sum().sort_values(ascending=True)
            fig = px.bar(stock_by_model,
                        orientation='h',
                        title='Current Stock by Model',
                        labels={'value': 'Units in Stock', 'Vehicle_Model': 'Model'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Stock turnover analysis
            if 'Inventory_Turnover_Ratio' in df.columns:
                turnover_by_model = df.groupby('Vehicle_Model')['Inventory_Turnover_Ratio'].mean().sort_values(ascending=True)
                fig = px.bar(turnover_by_model,
                            orientation='h',
                            title='Inventory Turnover by Model',
                            labels={'value': 'Turnover Ratio', 'Vehicle_Model': 'Model'})
                st.plotly_chart(fig, use_container_width=True)

def generate_context_from_df(df):
    """Generate focused context for data analysis"""
    context = {}
    
    try:
        # Get column names and data types
        context['columns'] = df.columns.tolist()
        context['data_types'] = df.dtypes.to_dict()
        
        # Get basic stats for numerical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        context['stats'] = df[numeric_cols].describe().to_dict()
        
        # Get unique values for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        context['categories'] = {col: df[col].unique().tolist() for col in categorical_cols}
        
        return str(context)  # Convert to string for prompt
    except Exception as e:
        return "Error processing data context"

def sales_chatbot(query: str, context: str) -> str:
    """
    Focused chatbot for data analysis and forecasting questions
    """
    prompt = f"""
You are a data analysis assistant. Your task is to answer questions about the CSV data.
Available data context: {context}

User Question: {query}

Rules:
1. Only answer questions about the data and forecasting
2. Provide maximum 2 sentences
3. If you can't answer from the data, say "I cannot answer that from the available data"
4. Focus on numerical insights and trends
5. Don't mention the context or data structure in your response

Question: {query}
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([prompt])
        response_text = response.text.strip()
        # Limit to two sentences
        sentences = response_text.split('. ')
        return '. '.join(sentences[:2]) + ('.' if not response_text.endswith('.') else '')
    except Exception:
        return "I cannot process that request at the moment."

def display_chat(context_summary):
    """Display focused chat interface"""
    st.markdown("## 💬 Data Analysis Assistant")
    chat.display_chat(context_summary)

def analyze_uploaded_data(df):
    """Analyze newly uploaded data using LLM"""
    try:
        context = generate_context_from_df(df)
        initial_analysis_prompt = """
        Analyze this dataset and provide:
        1. Key insights about the data
        2. Important trends
        3. Any anomalies or concerns
        Keep the response concise and focused on the most important findings.
        """
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([f"Context: {context}\n\nTask: {initial_analysis_prompt}"])
        return response.text.strip()
    except Exception as e:
        return "Could not analyze the data at this moment."

def main():
    init_session_state()
    
    if 'initial_analyses' not in st.session_state:
        st.session_state.initial_analyses = {}
    
    if 'chat_context' not in st.session_state:
        st.session_state.chat_context = ""

    st.title("📊 Business Intelligence Dashboard")
    
    # File Upload Section
    with st.expander("📁 Upload Data Files", expanded=False):
        new_files = st.file_uploader(
            "Upload your inventory and sales data files", 
            type="csv", 
            accept_multiple_files=True, 
            key="main_file_uploader"
        )
        
        if new_files:
            for nf in new_files:
                if all(nf.name != f.name for f in st.session_state.uploaded_files):
                    st.session_state.uploaded_files.append(nf)
                    df = utils.read_file(nf)
                    if df is not None:
                        st.session_state.file_dfs[nf.name] = df
                        # Generate context for chat when new file is uploaded
                        file_context = generate_context_from_df(df)
                        st.session_state.chat_context += f"\nFile: {nf.name}\n{file_context}\n---\n"
                        analysis = analyze_uploaded_data(df)
                        st.session_state.initial_analyses[nf.name] = analysis

\
    chat_col1, chat_col2 = st.columns([2, 1])
    
    with chat_col1:
        display_chat(st.session_state.chat_context)  # Pass the accumulated context
    
    with chat_col2:
        st.markdown("### 🤖 Chat Tips")
        st.info("""
        Try asking:
        - "What are the current sales trends?"
        - "Show top performing products"
        - "What's the total revenue?"
        - "Which product has the highest inventory?"
        - "Compare sales between different models"
        - "Show me the seasonal patterns"
        """)

    # Display analyses and visualizations
    if st.session_state.file_dfs:
        for file_name, df in st.session_state.file_dfs.items():
            st.markdown(f"## 📊 Analysis for {file_name}")
            
            # KPI section
            kpi.create_kpi_section(df)
            
            # Sales Analysis
            st.markdown("---")
            sales.display_sales_trends(df)
            
            # Inventory Insights
            st.markdown("---")
            display_inventory_insights(df)
            
            # Predictive Analytics
            model, metrics = predictive.train_model(df)
            if model and metrics:
                with st.container():
                    predictive.display_predictive_analytics(df, model, metrics)
                # predictive.display_model_performance(metrics) # Commented out to remove model performance
        
        # Initial Analysis Section - Moved to bottom
        st.markdown("---")
        st.markdown("## 📝 Initial Data Analysis")
        for file_name, analysis in st.session_state.initial_analyses.items():
            with st.expander(f"AI Analysis for {file_name}", expanded=True):
                st.markdown(analysis)
                if st.button(f"🔄 Refresh Analysis for {file_name}"):
                    df = st.session_state.file_dfs[file_name]
                    new_analysis = analyze_uploaded_data(df)
                    st.session_state.initial_analyses[file_name] = new_analysis
                    st.rerun()

if __name__ == "__main__":
    main()
