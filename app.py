# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import numpy as np
from config import CONFIG
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.predictor import Predictor
from utils.helpers import save_uploaded_file, load_brand_data
import pytz


class StockDataProcessor:
    def __init__(self):
        self.trading_hours = {
            'start': '09:00',  # Market opens at 9:00 AM
            'end': '15:00'  # Market closes at 3:00 PM
        }

    def is_weekday(self, date):
        """Check if date is a weekday (Monday=0 to Friday=4)"""
        return date.weekday() < 5

    def get_next_trading_day(self, date):
        """Get next trading day (skip weekends)"""
        next_day = date + timedelta(days=1)
        while not self.is_weekday(next_day):
            next_day += timedelta(days=1)
        return next_day

    def filter_weekdays_only(self, df):
        """Filter dataframe to only include weekdays"""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Keep only Monday (0) to Friday (4)
            weekday_mask = df['timestamp'].dt.dayofweek < 5
            return df[weekday_mask].copy()
        return df

    def generate_hourly_prices(self, open_price, close_price, high, low, timestamp):
        """Generate synthetic hourly prices with increased randomness and market dynamics"""
        # Only generate for weekdays
        if not self.is_weekday(timestamp.date()):
            return [], []

        num_points = 6  # Only 6 points from 9 AM to 3 PM
        np.random.seed(int(timestamp.timestamp()))  # Seed for reproducibility

        # Create price trajectory with more dynamic variation
        hours = np.linspace(9, 15, num=num_points)  # From 9 AM to 3 PM
        prices = np.zeros(num_points)

        # Start and end points
        prices[0] = open_price
        prices[-1] = close_price

        # Price range and volatility
        price_range = high - low
        volatility = price_range * 0.15  # Increased volatility to 15%

        # Market session dynamics
        def market_hour_effect(hour):
            """Simulate typical market hour price movements"""
            # More volatility in market opening and closing hours
            market_hours = [(0, 2), (9, 11), (21, 22)]  # Early morning, mid-morning, late afternoon
            for start, end in market_hours:
                if start <= hour <= end:
                    return 1.5  # Higher volatility during these periods
            return 1.0

        # Generate intermediate prices
        for i in range(1, num_points - 1):
            # Progress through trading day
            progress = (hours[i] - 9) / 6  # Scale to progress from 9 to 15 hours

            # Base interpolation between open and close with non-linear progression
            base_price = open_price * (1 - np.sin(progress * np.pi)) + close_price * np.sin(progress * np.pi)

            # Market hour effect
            hour_effect = market_hour_effect(hours[i])

            # Add randomness with different distribution
            price_noise = np.random.normal(0, volatility * hour_effect) * np.sin(progress * np.pi)
            momentum = np.random.uniform(-volatility, volatility) * hour_effect
            trend_component = (close_price - open_price) * progress * 0.5

            # Calculate price with multiple randomness factors
            new_price = base_price + price_noise + momentum + trend_component

            # Constrain within day's range with some flex
            new_price = max(min(new_price, high * 1.02), low * 0.98)
            prices[i] = new_price

        # Generate timestamps for the selected time range
        tz = pytz.timezone("Asia/Karachi")
        base_time = timestamp.replace(hour=9, minute=0, second=0, microsecond=0).astimezone(tz)

        timestamps = [base_time + timedelta(hours=h - 9) for h in hours]  # Adjust for 9 AM as start time

        return timestamps, prices

    def process_stock_data(self, df):
        """Convert daily OHLC data to hourly data with interpolation and increased realism"""
        # Ensure datetime conversion
        df['timestamp'] = pd.to_datetime(df['Date']).dt.tz_localize('UTC').dt.tz_convert('Asia/Karachi')
        df = df.sort_values('timestamp')

        # Filter out weekends from input data
        df = self.filter_weekdays_only(df)

        # Container for hourly data
        hourly_data = []

        # Process each unique date separately (only weekdays now)
        for date in df['Date'].unique():
            # Get row for specific date
            row = df[df['Date'] == date].iloc[0]

            # Skip if it's a weekend (double check)
            if not self.is_weekday(pd.to_datetime(date).date()):
                continue

            # Get prices and timestamp
            open_price = float(row['Open'])
            close_price = float(row['Price'])
            high_price = float(row['High'])
            low_price = float(row['Low'])
            timestamp = row['timestamp']

            # Generate hourly timestamps and prices
            hourly_timestamps, hourly_prices = self.generate_hourly_prices(
                open_price, close_price, high_price, low_price, timestamp
            )

            # Create hourly records
            for t, v in zip(hourly_timestamps, hourly_prices):
                hourly_data.append({
                    'timestamp': t,
                    'value': v,
                    'original_date': date,
                    'is_market_hour': True
                })

        # Create DataFrame with generated hourly data
        hourly_df = pd.DataFrame(hourly_data)
        if not hourly_df.empty:
            hourly_df = hourly_df.sort_values('timestamp')
            # Add technical indicators
            hourly_df = self.add_technical_indicators(hourly_df)

        return hourly_df

    def add_technical_indicators(self, df):
        """Add technical indicators to the hourly data"""
        if df.empty:
            return df

        # Simple Moving Averages
        df['SMA_5'] = df['value'].rolling(window=5).mean()
        df['SMA_20'] = df['value'].rolling(window=20).mean()

        # Relative Strength Index (RSI)
        delta = df['value'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_middle'] = df['value'].rolling(window=20).mean()
        bb_std = df['value'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)

        # MACD
        exp1 = df['value'].ewm(span=12, adjust=False).mean()
        exp2 = df['value'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        return df


def prepare_stock_data(csv_file):
    """Prepare stock data from CSV file"""
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Initialize processor
    processor = StockDataProcessor()

    # Process data
    hourly_df = processor.process_stock_data(df)

    return hourly_df


def create_historical_graph(df, predictions, brand, start_date, end_date):
    """Create historical graph with future predictions for selected date range (weekdays only)"""
    fig = go.Figure()

    # Filter data for selected date range and weekdays only
    mask = ((df['timestamp'].dt.date >= start_date) &
            (df['timestamp'].dt.date <= end_date) &
            (df['timestamp'].dt.dayofweek < 5))  # Only weekdays
    filtered_df = df[mask]

    # Filter predictions for weekdays only
    if 'daily' in predictions and not predictions['daily'].empty:
        pred_mask = ((predictions['daily']['timestamp'].dt.date >= start_date) &
                     (predictions['daily']['timestamp'].dt.date <= end_date) &
                     (predictions['daily']['timestamp'].dt.dayofweek < 5))  # Only weekdays
        filtered_predictions = predictions['daily'][pred_mask]
    else:
        filtered_predictions = pd.DataFrame()

    # Historical data
    if not filtered_df.empty:
        fig.add_trace(go.Scatter(
            x=filtered_df['timestamp'],
            y=filtered_df['value'],
            name='Historical Data',
            line=dict(color='blue', width=2, shape='spline'),
            mode='lines'
        ))

    # Future predictions
    if not filtered_predictions.empty:
        fig.add_trace(go.Scatter(
            x=filtered_predictions['timestamp'],
            y=filtered_predictions['predicted_value'],
            name='Future Predictions',
            line=dict(color='red', width=2, dash='dash', shape='spline'),
            mode='lines'
        ))

    fig.update_layout(
        title=dict(
            text=f"{brand} - Historical Data and Predictions ({start_date} to {end_date}) - Weekdays Only",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Date",
            gridcolor='lightgrey',
            showgrid=True
        ),
        yaxis=dict(
            title="Value",
            gridcolor='lightgrey',
            showgrid=True
        ),
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        plot_bgcolor='white'
    )

    return fig
def create_daily_prediction_graph(df, predictions, selected_date, brand):
    """Create daily prediction graph with LINES and weekday logic"""
    fig = go.Figure()
    processor = StockDataProcessor()

    # Convert selected_date to pandas Timestamp for consistent comparison
    if isinstance(selected_date, str):
        selected_date = pd.to_datetime(selected_date).date()
    elif hasattr(selected_date, 'date'):
        selected_date = selected_date.date()

    # Skip if selected date is weekend
    if not processor.is_weekday(selected_date):
        fig.add_annotation(
            text="Market is closed on weekends. Please select a weekday.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig

    # Ensure timestamps are properly formatted
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'daily' in predictions and not predictions['daily'].empty:
        predictions['daily']['timestamp'] = pd.to_datetime(predictions['daily']['timestamp'])

    # Filter actual data for the selected date (weekdays only)
    daily_data = df[(df['timestamp'].dt.date == selected_date) &
                    (df['timestamp'].dt.dayofweek < 5)].copy()

    # Get the latest available data date
    latest_data_date = df['timestamp'].dt.date.max()

    # Determine prediction date and filter predictions
    daily_predictions = pd.DataFrame()
    prediction_label = "Predictions"

    if 'daily' in predictions and not predictions['daily'].empty:
        if selected_date <= latest_data_date:
            # Historical day - show predictions for next TRADING day
            prediction_date = processor.get_next_trading_day(selected_date)
            daily_predictions = predictions['daily'][
                (predictions['daily']['timestamp'].dt.date == prediction_date) &
                (predictions['daily']['timestamp'].dt.dayofweek < 5)  # Ensure weekday
                ].copy()
            prediction_label = f"Predictions for {prediction_date}"
        else:
            # Future day - show predictions for that day (if weekday)
            if processor.is_weekday(selected_date):
                daily_predictions = predictions['daily'][
                    (predictions['daily']['timestamp'].dt.date == selected_date) &
                    (predictions['daily']['timestamp'].dt.dayofweek < 5)
                    ].copy()
                prediction_label = f"Predictions for {selected_date}"

    # Add actual data trace - LINES WITH MARKERS
    if not daily_data.empty:
        fig.add_trace(
            go.Scatter(
                x=daily_data['timestamp'],
                y=daily_data['value'],
                name='Actual Data',
                mode='lines+markers',  # LINES WITH MARKERS
                line=dict(color='royalblue', width=2),
                marker=dict(size=8, color='royalblue', symbol='circle'),
                hovertemplate='<b>%{x|%H:%M}</b><br>Actual: <b>%{y:.2f}</b><extra></extra>',
            )
        )

    # Add prediction trace - LINES WITH MARKERS
    if not daily_predictions.empty:
        fig.add_trace(
            go.Scatter(
                x=daily_predictions['timestamp'],
                y=daily_predictions['predicted_value'],
                name=prediction_label,
                mode='lines+markers',  # LINES WITH MARKERS
                line=dict(color='firebrick', width=2, dash='dash'),  # Dashed line for predictions
                marker=dict(size=8, symbol='diamond', color='firebrick'),
                hovertemplate='<b>%{x|%H:%M}</b><br>Predicted: <b>%{y:.2f}</b><extra></extra>',
            )
        )

    # Create title
    weekday_name = pd.to_datetime(selected_date).strftime('%A')
    title = f"{brand} - Daily View ({weekday_name}, {selected_date}) - Trading Hours Only"
    if selected_date <= latest_data_date and not daily_predictions.empty:
        next_day = processor.get_next_trading_day(selected_date)
        next_weekday = pd.to_datetime(next_day).strftime('%A')
        title += f" + {next_weekday} Predictions"

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, family="Arial, sans-serif"),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Time (Trading Hours: 9 AM - 3 PM)",
            tickformat='%H:%M',
            dtick=3600000,  # 1 hour intervals
            gridcolor='lightgrey',
            showgrid=True,
            tickangle=45,
            range=[
                pd.to_datetime(f"{selected_date} 08:30:00"),
                pd.to_datetime(f"{selected_date} 15:30:00")
            ]
        ),
        yaxis=dict(
            title="Value",
            gridcolor='lightgrey',
            showgrid=True
        ),
        hovermode='x unified',
        height=450,
        plot_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=dict(l=50, r=50, t=80, b=80)
    )

    # Add annotations if no data is available
    if daily_data.empty and daily_predictions.empty:
        fig.add_annotation(
            text="No trading data available for the selected date",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
    elif daily_data.empty:
        fig.add_annotation(
            text="No actual trading data available for this date",
            xref="paper", yref="paper",
            x=0.5, y=0.1,
            showarrow=False,
            font=dict(size=12, color="gray")
        )
    elif daily_predictions.empty:
        fig.add_annotation(
            text="No predictions available for the next trading day",
            xref="paper", yref="paper",
            x=0.5, y=0.1,
            showarrow=False,
            font=dict(size=12, color="gray")
        )

    return fig

def create_hourly_prediction_graph(df, predictions, selected_date, brand):
    """Create hourly prediction graph with smooth lines (weekdays only)"""
    fig = go.Figure()
    processor = StockDataProcessor()

    # Convert selected_date to proper format
    if isinstance(selected_date, str):
        selected_date = pd.to_datetime(selected_date)
    elif not hasattr(selected_date, 'date'):
        selected_date = pd.to_datetime(selected_date)

    # Skip if selected date is weekend
    if not processor.is_weekday(selected_date.date()):
        fig.add_annotation(
            text="Market is closed on weekends. Please select a weekday.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig

    # Ensure consistent timezone
    tz = pytz.timezone('Asia/Karachi')
    if selected_date.tzinfo is None:
        selected_date = selected_date.tz_localize(tz)

    # Convert timestamp columns to timezone-aware datetimes
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('Asia/Karachi')

    if 'hourly' in predictions and not predictions['hourly'].empty:
        predictions['hourly']['timestamp'] = pd.to_datetime(predictions['hourly']['timestamp'])
        if predictions['hourly']['timestamp'].dt.tz is None:
            predictions['hourly']['timestamp'] = predictions['hourly']['timestamp'].dt.tz_localize('Asia/Karachi')

    # Filter data for selected date using date comparison (weekdays only)
    hourly_data = df[(df['timestamp'].dt.date == selected_date.date()) &
                     (df['timestamp'].dt.dayofweek < 5)]

    if 'hourly' in predictions and not predictions['hourly'].empty:
        hourly_predictions = predictions['hourly'][
            (predictions['hourly']['timestamp'].dt.date == selected_date.date()) &
            (predictions['hourly']['timestamp'].dt.dayofweek < 5)
            ]
    else:
        hourly_predictions = pd.DataFrame()

    # Filter the hourly data to only include times between 9 AM and 3 PM
    hourly_data = hourly_data[(hourly_data['timestamp'].dt.hour >= 9) & (hourly_data['timestamp'].dt.hour <= 15)]
    if not hourly_predictions.empty:
        hourly_predictions = hourly_predictions[
            (hourly_predictions['timestamp'].dt.hour >= 9) & (hourly_predictions['timestamp'].dt.hour <= 15)]

    # Ensure we sample exactly 6 points if available
    if len(hourly_data) >= 6:
        hourly_data = hourly_data.iloc[np.linspace(0, len(hourly_data) - 1, 6, dtype=int)]
    if len(hourly_predictions) >= 6:
        hourly_predictions = hourly_predictions.iloc[np.linspace(0, len(hourly_predictions) - 1, 6, dtype=int)]

    # Add actual data trace
    if not hourly_data.empty:
        fig.add_trace(go.Scatter(
            x=hourly_data['timestamp'],
            y=hourly_data['value'],
            name='Actual Data',
            line=dict(color='blue', width=3, shape='spline'),
            mode='lines+markers',
            marker=dict(size=8, color='blue')
        ))

    # Add predictions trace
    if not hourly_predictions.empty:
        fig.add_trace(go.Scatter(
            x=hourly_predictions['timestamp'],
            y=hourly_predictions['predicted_value'],
            name='Predictions',
            line=dict(color='red', width=3, dash='dash', shape='spline'),
            mode='lines+markers',
            marker=dict(size=8, symbol='diamond', color='red')
        ))

    weekday_name = selected_date.strftime('%A')
    fig.update_layout(
        title=dict(
            text=f"{brand} - Hourly Predictions ({weekday_name}, {selected_date.date()}) - Trading Hours",
            font=dict(size=18)
        ),
        xaxis=dict(
            title="Time (9 AM - 3 PM Trading Hours)",
            tickformat='%H:%M',
            dtick=3600000,  # 1 hour in milliseconds
            gridcolor='lightgrey',
            showgrid=True
        ),
        yaxis=dict(
            title="Value",
            gridcolor='lightgrey',
            showgrid=True
        ),
        height=450,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        plot_bgcolor='white'
    )

    # Add annotation if no data
    if hourly_data.empty and hourly_predictions.empty:
        fig.add_annotation(
            text="No hourly trading data available for the selected date",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )

    return fig


def get_available_weekdays(start_date, end_date):
    """Get list of available weekdays between start and end date"""
    weekdays = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday=0 to Friday=4
            weekdays.append(current_date)
        current_date += timedelta(days=1)
    return weekdays


def main():
    st.set_page_config(layout="wide", page_title="Brand Prediction Dashboard - Trading Days Only")
    st.title("📈 Brand Prediction Dashboard - Trading Days Only")
    st.markdown("*Market operates Monday-Friday, 9 AM - 3 PM (Pakistan Time)*")

    # Initialize classes
    data_processor = DataProcessor()
    model_trainer = ModelTrainer(CONFIG['MODELS_DIR'])
    stock_processor = StockDataProcessor()

    # Sidebar layout
    st.sidebar.header("Settings")
    selected_brand = st.sidebar.selectbox("Select Brand", CONFIG['AVAILABLE_BRANDS'])

    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        f"Upload new data for {selected_brand}",
        type=['csv']
    )

    try:
        if uploaded_file:
            st.info("Processing uploaded file...")
            df = save_uploaded_file(uploaded_file, selected_brand, CONFIG['DATA_DIR'])

            if df is not None:
                # Filter weekdays only during processing
                processed_df = data_processor.process_data(df)
                processed_df = stock_processor.filter_weekdays_only(processed_df)

                hourly_model, hourly_scaler = model_trainer.train_model(processed_df, selected_brand, 'hourly')
                daily_model, daily_scaler = model_trainer.train_model(processed_df, selected_brand, 'daily')
                st.success(f"Model trained successfully for {selected_brand} (weekdays only)")

        # Load data and models
        df = load_brand_data(selected_brand, CONFIG['DATA_DIR'])
        hourly_model, hourly_scaler = model_trainer.load_model(selected_brand, 'hourly')
        daily_model, daily_scaler = model_trainer.load_model(selected_brand, 'daily')

        if df is not None and hourly_model is not None and daily_model is not None:
            processed_df = data_processor.process_data(df)
            # Filter to weekdays only
            processed_df = stock_processor.filter_weekdays_only(processed_df)

            # Date range selection in sidebar
            st.sidebar.header("Date Range Selection")
            min_date = processed_df['timestamp'].dt.date.min()
            max_date = processed_df['timestamp'].dt.date.max()

            # Date range selection with two columns
            date_cols = st.sidebar.columns(2)
            with date_cols[0]:
                start_date = st.date_input(
                    "Start Date",
                    value=max_date - timedelta(days=7),
                    min_value=min_date,
                    max_value=max_date
                )

            with date_cols[1]:
                end_date = st.date_input(
                    "End Date",
                    value=max_date,
                    min_value=start_date,
                    max_value=max_date + timedelta(days=30)
                )

            if start_date > end_date:
                st.error("Error: End date must be after start date")
                return

            # Get available weekdays for selection
            available_weekdays = get_available_weekdays(start_date, end_date)

            if available_weekdays:
                # Selected date for daily/hourly view (weekdays only)
                weekday_options = {date.strftime('%A, %Y-%m-%d'): date for date in available_weekdays}
                selected_date_str = st.sidebar.selectbox(
                    "Select Trading Day for Daily/Hourly View",
                    options=list(weekday_options.keys()),
                    index=len(weekday_options) - 1  # Default to last available weekday
                )
                selected_date = weekday_options[selected_date_str]
            else:
                st.sidebar.error("No trading days found in the selected range")
                return

            # Initialize predictor
            predictor = Predictor(
                hourly_model, daily_model,
                hourly_scaler, daily_scaler,
                data_processor
            )

            # Generate predictions
            last_timestamp = processed_df['timestamp'].max()
            predictions = predictor.predict(last_timestamp, 'both')

            # Filter predictions to weekdays only
            if 'daily' in predictions and not predictions['daily'].empty:
                predictions['daily'] = stock_processor.filter_weekdays_only(predictions['daily'])
            if 'hourly' in predictions and not predictions['hourly'].empty:
                predictions['hourly'] = stock_processor.filter_weekdays_only(predictions['hourly'])

            # Debug information
            with st.expander("📊 Debug Information"):
                st.write(f"**Processed DF shape:** {processed_df.shape}")
                st.write(
                    f"**Date range in data:** {processed_df['timestamp'].dt.date.min()} to {processed_df['timestamp'].dt.date.max()}")
                st.write(f"**Trading days in data:** {len(processed_df['timestamp'].dt.date.unique())} days")
                if 'daily' in predictions and not predictions['daily'].empty:
                    st.write(f"**Daily predictions shape:** {predictions['daily'].shape}")
                else:
                    st.write("**Daily predictions:** Not available")
                if 'hourly' in predictions and not predictions['hourly'].empty:
                    st.write(f"**Hourly predictions shape:** {predictions['hourly'].shape}")
                else:
                    st.write("**Hourly predictions:** Not available")

            # Filter data for selected date range (weekdays only)
            mask = ((processed_df['timestamp'].dt.date >= start_date) &
                    (processed_df['timestamp'].dt.date <= end_date) &
                    (processed_df['timestamp'].dt.dayofweek < 5))
            date_range_df = processed_df[mask]

            # Display metrics for selected range
            st.subheader(f"📈 Trading Metrics ({start_date} to {end_date})")
            if not date_range_df.empty:
                metric_cols = st.columns(5)

                with metric_cols[0]:
                    range_avg = date_range_df['value'].mean()
                    st.metric("Average Value", f"{range_avg:.2f}")

                with metric_cols[1]:
                    range_max = date_range_df['value'].max()
                    st.metric("Maximum Value", f"{range_max:.2f}")

                with metric_cols[2]:
                    range_min = date_range_df['value'].min()
                    st.metric("Minimum Value", f"{range_min:.2f}")

                with metric_cols[3]:
                    if len(date_range_df) > 1:
                        value_change = date_range_df['value'].iloc[-1] - date_range_df['value'].iloc[0]
                        st.metric("Value Change", f"{value_change:.2f}")
                    else:
                        st.metric("Value Change", "N/A")

                with metric_cols[4]:
                    trading_days = len(date_range_df['timestamp'].dt.date.unique())
                    st.metric("Trading Days", f"{trading_days}")
            else:
                st.warning("No trading data available for the selected date range")

            # Display graphs
            st.subheader("📊 Historical Trading View")
            try:
                historical_fig = create_historical_graph(processed_df, predictions, selected_brand,
                                                         start_date, end_date)
                st.plotly_chart(historical_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating historical graph: {str(e)}")

            # Daily and Hourly Predictions
            st.subheader("🎯 Daily and Hourly Analysis")
            graph_cols = st.columns(2)

            with graph_cols[0]:
                st.subheader("📍 Daily View (Dots Only)")
                try:
                    # Check if we have data for the selected date
                    selected_date_data = processed_df[processed_df['timestamp'].dt.date == selected_date]
                    if selected_date_data.empty:
                        st.info(f"No actual trading data available for {selected_date}")

                    daily_fig = create_daily_prediction_graph(processed_df, predictions,
                                                              selected_date, selected_brand)
                    st.plotly_chart(daily_fig, use_container_width=True)

                    # Show next trading day info
                    if stock_processor.is_weekday(selected_date):
                        next_trading_day = stock_processor.get_next_trading_day(selected_date)
                        next_weekday_name = pd.to_datetime(next_trading_day).strftime('%A')
                        st.info(f"💡 Next trading day: {next_weekday_name}, {next_trading_day}")

                except Exception as e:
                    st.error(f"Error creating daily graph: {str(e)}")
                    st.write("Error details:", str(e))

            with graph_cols[1]:
                st.subheader("⏰ Hourly View (Trading Hours)")
                try:
                    hourly_fig = create_hourly_prediction_graph(processed_df, predictions,
                                                                selected_date, selected_brand)
                    st.plotly_chart(hourly_fig, use_container_width=True)

                    # Show trading hours info
                    st.info("📅 Trading Hours: 9:00 AM - 3:00 PM (Monday to Friday)")

                except Exception as e:
                    st.error(f"Error creating hourly graph: {str(e)}")
                    st.write("Error details:", str(e))

            # Weekend notification
            if selected_date.weekday() >= 5:  # Saturday or Sunday
                st.warning("⚠️ Selected date falls on a weekend. Market is closed on weekends.")
                next_monday = selected_date + timedelta(days=(7 - selected_date.weekday()))
                st.info(f"📅 Next trading day: Monday, {next_monday}")

            # Trading week summary
            st.subheader("📅 Trading Week Summary")

            # Get current week's data
            week_start = selected_date - timedelta(days=selected_date.weekday())
            week_end = week_start + timedelta(days=4)  # Friday

            week_data = processed_df[
                (processed_df['timestamp'].dt.date >= week_start) &
                (processed_df['timestamp'].dt.date <= week_end)
                ]

            if not week_data.empty:
                week_cols = st.columns(5)
                weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

                for i, day_name in enumerate(weekdays):
                    day_date = week_start + timedelta(days=i)
                    day_data = week_data[week_data['timestamp'].dt.date == day_date]

                    with week_cols[i]:
                        if not day_data.empty:
                            avg_value = day_data['value'].mean()
                            st.metric(f"{day_name[:3]}\n{day_date.strftime('%m-%d')}", f"{avg_value:.2f}")
                        else:
                            st.metric(f"{day_name[:3]}\n{day_date.strftime('%m-%d')}", "No Data")

        else:
            st.warning(f"No data available for {selected_brand}. Please upload data.")

            # Show what's missing
            if df is None:
                st.error("❌ No data file found")
            if hourly_model is None:
                st.error("❌ No hourly model found")
            if daily_model is None:
                st.error("❌ No daily model found")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please ensure your CSV file contains at least two columns: a date/time column and a value column.")

        # Show full error details in expander
        with st.expander("Full Error Details"):
            import traceback
            st.code(traceback.format_exc())

    # Footer with trading information
    st.markdown("---")
    st.markdown("""
    **🏢 Trading Information:**
    - **Market Days:** Monday to Friday only
    - **Trading Hours:** 9:00 AM - 3:00 PM (Pakistan Standard Time)
    - **Weekend Policy:** Market closed on Saturday & Sunday
    - **Prediction Logic:** Friday predictions show Monday's forecast (skipping weekend)
    """)


if __name__ == "__main__":
    main()