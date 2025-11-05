"""
Utils package
Contains utility functions for text processing, sentiment analysis, 
visualization, and email services
"""

from .text_processing import clean_text, batch_clean_texts, get_top_words, get_text_statistics
from .sentiment_models import (
    textblob_sentiment,
    textblob_sequential,
    textblob_parallel,
    load_roberta_model,
    roberta_sentiment_single,
    roberta_sequential,
    roberta_parallel
)
from .visualization import (
    plot_sentiment_distribution,
    plot_top_words,
    plot_confusion_matrix,
    plot_time_comparison,
    plot_model_comparison,
    create_metrics_cards,
    plot_polarity_distribution,
    plot_confidence_distribution
)
from .email_service import send_results_email, validate_email_config, get_email_setup_instructions

__all__ = [
    # Text processing
    'clean_text',
    'batch_clean_texts',
    'get_top_words',
    'get_text_statistics',
    
    # Sentiment models
    'textblob_sentiment',
    'textblob_sequential',
    'textblob_parallel',
    'load_roberta_model',
    'roberta_sentiment_single',
    'roberta_sequential',
    'roberta_parallel',
    
    # Visualization
    'plot_sentiment_distribution',
    'plot_top_words',
    'plot_confusion_matrix',
    'plot_time_comparison',
    'plot_model_comparison',
    'create_metrics_cards',
    'plot_polarity_distribution',
    'plot_confidence_distribution',
    
    # Email service
    'send_results_email',
    'validate_email_config',
    'get_email_setup_instructions'
]