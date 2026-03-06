"""
Model Configuration

Defines all LLMs to be tested in this replication study.
Based on the model selection table from the research timeline.
"""

# Models organized by origin and type
MODELS = {
    # Western (Open Source)
    'western_open': {
        'gemma2:2b': {
            'provider': 'ollama',
            'origin': 'Google (US)',
            'size': '2B',
            'languages': ['English', 'multilingual'],
            'rationale': 'Stable Western open baseline; efficient and well-aligned',
            'priority': 'high'
        },
        'llama3.1:8b': {
            'provider': 'ollama',
            'origin': 'Meta (US)',
            'size': '8B',
            'languages': ['English', 'multilingual'],
            'rationale': 'High-capacity Western reference model',
            'priority': 'high'
        },
        'mistral:7b': {
            'provider': 'ollama',
            'origin': 'Mistral AI (EU)',
            'size': '7B',
            'languages': ['English', 'multilingual'],
            'rationale': 'Compact Western comparison with strong reasoning',
            'priority': 'high'
        },
    },

    # East Asian
    'east_asian': {
        'qwen2.5:1.5b': {
            'provider': 'ollama',
            'origin': 'Alibaba (China)',
            'size': '1.5B',
            'languages': ['Chinese', 'English', 'multilingual'],
            'rationale': 'Small Chinese anchor; efficient for large-N sweeps',
            'priority': 'high'
        },
        'qwen2.5:3b': {
            'provider': 'ollama',
            'origin': 'Alibaba (China)',
            'size': '3B',
            'languages': ['Chinese', 'English', 'multilingual'],
            'rationale': 'Mid-size Chinese anchor for size comparison',
            'priority': 'high'
        },
        'qwen2.5:7b': {
            'provider': 'ollama',
            'origin': 'Alibaba (China)',
            'size': '7B',
            'languages': ['Chinese', 'English', 'multilingual'],
            'rationale': 'Full-size Chinese anchor; primary non-Western model',
            'priority': 'high'
        },
        'yi:6b': {
            'provider': 'ollama',
            'origin': '01.AI (China)',
            'size': '6B',
            'languages': ['Chinese', 'English'],
            'rationale': 'Second Chinese-origin model; allows within-region comparison',
            'priority': 'medium'
        },
    },

    # MENA / Arabic
    'mena_arabic': {
        'salmatrafi/acegpt:7b': {
            'provider': 'ollama',
            'origin': 'MBZUAI (UAE)',
            'size': '7B',
            'languages': ['Arabic', 'English'],
            'rationale': 'Arabic-centric non-Western model; primary MENA region anchor',
            'priority': 'high'
        },
    },

    # Low-Resource / Global
    'global': {
        'phi3:mini': {
            'provider': 'ollama',
            'origin': 'Microsoft (US)',
            'size': '3.8B',
            'languages': ['English', 'multilingual'],
            'rationale': 'Efficient general model; robustness and CPU-sweep testing',
            'priority': 'high'
        },
    },

    # Proprietary (API-based)
    'proprietary': {
        'gpt-4o': {
            'provider': 'openai',
            'origin': 'OpenAI (US)',
            'size': 'unknown',
            'languages': ['English', 'multilingual'],
            'rationale': 'Primary replication target from PNAS paper',
            'priority': 'critical'
        },
        'gpt-4-turbo': {
            'provider': 'openai',
            'origin': 'OpenAI (US)',
            'size': 'unknown',
            'languages': ['English', 'multilingual'],
            'rationale': 'PNAS paper comparison model',
            'priority': 'high'
        },
        'gpt-4': {
            'provider': 'openai',
            'origin': 'OpenAI (US)',
            'size': 'unknown',
            'languages': ['English', 'multilingual'],
            'rationale': 'PNAS paper comparison model',
            'priority': 'high'
        },
        'gpt-3.5-turbo': {
            'provider': 'openai',
            'origin': 'OpenAI (US)',
            'size': 'unknown',
            'languages': ['English', 'multilingual'],
            'rationale': 'PNAS paper comparison model',
            'priority': 'medium'
        },
        'claude-sonnet-4-5': {
            'provider': 'anthropic',
            'origin': 'Anthropic (US)',
            'size': 'unknown',
            'languages': ['English', 'multilingual'],
            'rationale': 'Additional Western proprietary baseline',
            'priority': 'medium'
        },
        'gemini-1.5-pro': {
            'provider': 'google',
            'origin': 'Google (US)',
            'size': 'unknown',
            'languages': ['English', 'multilingual'],
            'rationale': 'Additional Western proprietary baseline',
            'priority': 'medium'
        },
    }
}


# Model testing parameters
DEFAULT_PARAMS = {
    'temperature': 0.0,  # Deterministic for baseline
    'max_tokens': 256,
    'top_p': 1.0,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0,
}

# For stochastic sensitivity testing (Week 7)
STOCHASTIC_PARAMS = {
    'temperature': 1.0,
    'seeds': list(range(10)),  # 10 different seeds
}


def get_models_by_priority(priority='high'):
    """
    Get list of models filtered by priority.

    Args:
        priority: 'critical', 'high', 'medium', or 'low'

    Returns:
        List of (model_name, model_config) tuples
    """
    priority_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
    threshold = priority_order.get(priority, 0)

    models = []
    for category, category_models in MODELS.items():
        for model_name, config in category_models.items():
            model_priority = priority_order.get(config.get('priority', 'low'), 1)
            if model_priority >= threshold:
                models.append((model_name, config))

    return models


def get_ollama_models():
    """Get list of models that should be run via Ollama."""
    models = []
    for _, category_models in MODELS.items():
        for model_name, config in category_models.items():
            if config.get('provider') == 'ollama':
                models.append(model_name)
    return models


def get_api_models():
    """Get list of models that require API access."""
    models = []
    for _, category_models in MODELS.items():
        for model_name, config in category_models.items():
            if config.get('provider') in ['openai', 'anthropic', 'google']:
                models.append((model_name, config['provider']))
    return models


def get_model_info(model_name):
    """Get configuration for a specific model."""
    for category, category_models in MODELS.items():
        if model_name in category_models:
            return category_models[model_name]
    return None


# Quick access to recommended models for each phase
RECOMMENDED_MODELS = {
    # Fast models for quick iteration and testing
    'testing': ['gemma2:2b', 'qwen2.5:1.5b'],
    # All currently installed local models
    'baseline': [
        'gemma2:2b', 'phi3:mini',
        'qwen2.5:1.5b', 'qwen2.5:3b', 'qwen2.5:7b',
        'mistral:7b', 'llama3.1:8b', 'yi:6b',
        'salmatrafi/acegpt:7b',
    ],
    # Baseline + proprietary API models
    'full': [
        'gemma2:2b', 'phi3:mini',
        'qwen2.5:1.5b', 'qwen2.5:3b', 'qwen2.5:7b',
        'mistral:7b', 'llama3.1:8b', 'yi:6b',
        'salmatrafi/acegpt:7b', 'gpt-4o',
    ],
}
