"""
Pages package
Contains all application pages
"""

from . import home
from . import upload
from . import cleaning
from . import analysis
from . import results
from . import email_results

__all__ = [
    'home',
    'upload',
    'cleaning',
    'analysis',
    'results',
    'email_results'
]