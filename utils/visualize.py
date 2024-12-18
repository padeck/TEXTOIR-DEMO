import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def scatterplot(filename):
    df = pd.read_json(f'../frontend/static/jsons/open_intent_detection/scatterplot_{filename}.json')
    df['text'] = df['text'].str.replace(r'\.', '.<br>', regex=True)
    df[['x', 'y']] = pd.DataFrame(df['coords'].to_list(), index=df.index)

    # Create the scatter plot using Plotly
    fig = px.scatter(df, x='x', y='y', color='pred', hover_data={'text': True, 'x': False, 'y': False, 'pred': True},
                    title="Scatter plot with hover text")

    # Customize layout
    fig.update_layout(
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate"
    )

    # Show the plot
    fig.show()

def detailed_error_analysis(filename):
    # Assuming you have your DataFrame ready (replace with actual data loading code)
    df_fine = pd.read_json('../frontend/static/jsons/open_intent_detection/true_false_fine.json')
    df_fine = df_fine[filename]
    df_fine = df_fine.to_frame()
    df_fine = df_fine.dropna()

    # Prepare labels and values
    labels = df_fine.index.tolist()
    values = [df_fine.loc[label][filename] for label in df_fine.index.tolist()]

    # Transpose the values for stacking (for correct stacking structure)
    values = np.array(values).T

    # Create the stacked bar chart (horizontal bars)
    fig = go.Figure()

    # Add each stack layer to the chart
    for i, row in enumerate(values):
        fig.add_trace(go.Bar(
            y=labels,  # Categories on the y-axis
            x=row,  # Values on the x-axis
            name=f'{labels[i]}',  # Each layer gets a specific name (Layer 1, Layer 2, etc.)
            orientation='h',  # Horizontal bars
            hovertemplate=(
                '<b>Intent: %{y}</b><br>'  # Show category (y-axis)
                'Predicted Label: %{customdata}<br>'  # Show the specific layer (customdata)
                'Count: %{x}<br>'  # Show the count for this layer (x value)
                '<extra></extra>'  # Remove default extra info (like trace name)
            ),
            customdata=[labels[i]] * len(row)  # Use customdata to store the layer index for hover
        ))

    # Update layout
    fig.update_layout(
        barmode='stack',  # Stack the bars
        title=f'Detailed Analysis of incorrectly predicted samples for {filename}',
        legend_title='Labels',
    )

    # Show the chart
    fig.show()

def fine_grained_performance(filename):
    # Load the JSON data
    df_bar = pd.read_json("../frontend/static/jsons/open_intent_detection/true_false_overall.json")

    # Sample data
    categories = df_bar[filename]['intent_class']
    values_left = df_bar[filename]['left']      # Negative values for left side
    values_right = df_bar[filename]['right']    # Positive values for right side

    # Create a horizontal bar chart with Plotly
    fig = go.Figure()

    # Add "Left" bars (negative)
    fig.add_trace(go.Bar(
        y=categories,            # Categories on the y-axis
        x=values_left,           # Negative values
        orientation='h',         # Horizontal bars
        name='Wrong',             # Legend name
        marker=dict(color='#E57373')  # Bar color
    ))

    # Add "Right" bars (positive)
    fig.add_trace(go.Bar(
        y=categories,
        x=values_right,          # Positive values
        orientation='h',
        name='Correct',
        marker=dict(color='#76C7C0')  # Bar color
    ))

    # Add a vertical line at x=0
    fig.add_vline(x=0, line=dict(color='black', width=1), name='Center Line')

    # Update layout for better appearance
    fig.update_layout(
        title="Fine grained performance",
        barmode='overlay',        # Overlay to ensure bars are centered properly
        legend=dict(title="Direction"),
        template="plotly",        # Use a clean theme
        height=600,               # Set a specific height for better appearance
        margin=dict(l=100, r=50, t=50, b=50)  # Adjust margins
    )

    # Show the interactive plot
    fig.show()
