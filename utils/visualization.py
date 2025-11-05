"""
Visualization Utilities
Charts, graphs, and confusion matrices
"""

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from streamlit.components.v1 import html
import plotly.graph_objects as go
def plot_sentiment_distribution(df, column, title):
    """
    Plot sentiment distribution as bar chart
    
    Args:
        df: DataFrame containing sentiment data
        column: Column name with sentiment labels
        title: Chart title
        
    Returns:
        Plotly figure
    """
    sentiment_counts = df[column].value_counts()
    
    colors = {
        'positive': '#10b981',
        'neutral': '#6b7280',
        'negative': '#ef4444'
    }
    
    fig = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        labels={'x': 'Sentiment', 'y': 'Count'},
        title=title,
        color=sentiment_counts.index,
        color_discrete_map=colors
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title="Sentiment",
        yaxis_title="Count",
        template="plotly_white"
    )
    
    return fig

def plot_top_words(top_words, title="Top 20 Most Frequent Words"):
    """
    Plot top words as horizontal bar chart
    
    Args:
        top_words: List of (word, count) tuples
        title: Chart title
        
    Returns:
        Plotly figure
    """
    words_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
    
    fig = px.bar(
        words_df,
        y='Word',
        x='Count',
        title=title,
        orientation='h',
        color='Count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        template="plotly_white"
    )
    
    return fig

def plot_confusion_matrix(y_true, y_pred, labels, title, normalize=False):
    """
    Plot confusion matrix using Plotly
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        title: Chart title
        normalize: Whether to normalize values
        
    Returns:
        Plotly figure
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        text = [[f'{val:.1f}%' for val in row] for row in cm_display]
    else:
        cm_display = cm
        text = [[str(val) for val in row] for row in cm_display]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_display,
        x=labels,
        y=labels,
        text=text,
        texttemplate='%{text}',
        colorscale='Blues',
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Value: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=600,
        height=600
    )
    
    return fig

def plot_time_comparison(time_data):
    """
    Plot processing time comparison
    
    Args:
        time_data: Dictionary with method names and times
        
    Returns:
        Plotly figure
    """
    methods = list(time_data.keys())
    times = list(time_data.values())
    
    # Format method names
    formatted_methods = [m.replace('_', ' ').title() for m in methods]
    
    fig = px.bar(
        x=formatted_methods,
        y=times,
        title='Processing Time Comparison',
        labels={'x': 'Method', 'y': 'Time (seconds)'},
        color=times,
        color_continuous_scale='RdYlGn_r'
    )
    
    fig.update_traces(
        text=[f'{t:.2f}s' for t in times],
        textposition='outside'
    )
    
    fig.update_layout(
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

def plot_model_comparison(textblob_dist, roberta_dist):
    """
    Plot side-by-side sentiment distribution comparison
    
    Args:
        textblob_dist: TextBlob sentiment distribution
        roberta_dist: RoBERTa sentiment distribution
        
    Returns:
        Plotly figure
    """
    labels = ['Positive', 'Neutral', 'Negative']
    
    fig = go.Figure(data=[
        go.Bar(name='TextBlob', x=labels, y=textblob_dist, marker_color='#667eea'),
        go.Bar(name='RoBERTa', x=labels, y=roberta_dist, marker_color='#764ba2')
    ])
    
    fig.update_layout(
        title='Model Comparison: Sentiment Distribution',
        xaxis_title='Sentiment',
        yaxis_title='Count',
        barmode='group',
        template="plotly_white"
    )
    
    return fig

from streamlit.components.v1 import html

def create_metrics_cards(metrics: dict):
    """Render beautiful metric cards using HTML components."""
    
    card_html = """
    <style>
    .metric-wrapper {
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        margin-top: 0.5rem;
    }
    .metric-card {
        flex: 1;
        min-width: 180px;
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        border-radius: 16px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
        padding: 1.2rem;
        border-left: 6px solid;
        transition: all 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    .metric-title {
        margin: 0;
        color: #6b7280;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .metric-value {
        margin: 0.5rem 0 0 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    </style>
    <div class='metric-wrapper'>
    """

    icons = ['📄', '🧮', '🔤', '🪶', '📊']
    colors = ['#764ba2', '#10b981', '#f59e0b', '#3b82f6', '#ef4444']

    for i, (key, val) in enumerate(metrics.items()):
        color = colors[i % len(colors)]
        icon = icons[i % len(icons)]
        card_html += f"""
        <div class='metric-card' style='border-left-color:{color};'>
            <p class='metric-title'>{icon} {key}</p>
            <p class='metric-value' style='color:{color};'>{val}</p>
        </div>
        """
    card_html += "</div>"

    # Use the Streamlit HTML component (always renders correctly)
    html(card_html, height=220)
    return ""



def plot_polarity_distribution(polarities, title="Sentiment Polarity Distribution"):
    """
    Plot polarity score distribution as histogram
    
    Args:
        polarities: List of polarity scores
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = px.histogram(
        x=polarities,
        nbins=50,
        title=title,
        labels={'x': 'Polarity Score', 'y': 'Count'},
        color_discrete_sequence=['#667eea']
    )
    
    fig.update_layout(
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

def plot_confidence_distribution(confidences, title="Model Confidence Distribution"):
    """
    Plot confidence score distribution
    
    Args:
        confidences: List of confidence scores
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = px.histogram(
        x=confidences,
        nbins=50,
        title=title,
        labels={'x': 'Confidence Score', 'y': 'Count'},
        color_discrete_sequence=['#764ba2']
    )
    
    fig.update_layout(
        showlegend=False,
        template="plotly_white"
    )
    
    return fig
def plot_text_statistics(stats_dict, title="Text Statistics"):
    """
    Create a simple Plotly bar chart of text statistics.
    
    Args:
        stats_dict (dict): Dictionary of statistics (e.g., avg words, total texts, etc.)
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure
    """
    labels = list(stats_dict.keys())
    values = list(stats_dict.values())

    fig = go.Figure([go.Bar(x=labels, y=values, text=[f"{v:.1f}" if isinstance(v, (float, int)) else v for v in values],
                            textposition="auto")])
    fig.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title="Value",
        template="plotly_white",
        height=400,
        margin=dict(t=50, l=50, r=30, b=50)
    )
    return fig