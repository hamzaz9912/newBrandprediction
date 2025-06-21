def create_daily_prediction_graph(processed_df, predictions, selected_date, brand_name):
    selected_date = pd.to_datetime(selected_date).date()

    # Filter for the selected date
    actual = processed_df[processed_df['Datetime'].dt.date == selected_date]
    predicted = predictions[predictions['Datetime'].dt.date == selected_date]

    fig = go.Figure()

    # Actual values
    fig.add_trace(go.Scatter(
        x=actual['Datetime'], y=actual['Price'],
        mode='lines+markers', name='Actual Price',
        line=dict(color='blue')
    ))

    # Predicted values
    fig.add_trace(go.Scatter(
        x=predicted['Datetime'], y=predicted['Predicted Price'],
        mode='lines+markers', name='Predicted Price',
        line=dict(color='orange')
    ))

    fig.update_layout(
        title=f"{brand_name} - Daily Prediction ({selected_date})",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white"
    )

    return fig
