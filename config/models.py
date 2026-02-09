"""
Model Configuration

Defines all LLMs to be tested in this replication study.
Based on the model selection table from the research timeline.
"""

# Models organized by origin and type
MODELS = {
    # Western (Open Source)
    'western_open': {
        'gemma-2-2b-instruct': {
            'provider': 'ollama',
            'origin': 'Google (US)',
            'size': '2B',
            'languages': ['English', 'multilingual'],
            'rationale': 'Stable Western open baseline; efficient and well-aligned',
            'quantization': None,
            'priority': 'high'
        },
        'llama3-8b-instruct': {
            'provider': 'ollama',
            'origin': 'Meta (US)',
            'size': '8B',
            'languages': ['English', 'multilingual'],
            'rationale': 'High-capacity Western reference model',
            'quantization': '4-bit',
            'priority': 'high',
            'note': 'Limited runs due to size'
        },
        'mistral-7b-instruct': {
            'provider': 'ollama',
            'origin': 'Mistral AI (EU)',
            'size': '7B',
            'languages': ['English', 'multilingual'],
            'rationale': 'Compact Western comparison with strong reasoning',
            'quantization': 'quantized',
            'priority': 'medium'
        },
    },

    # East Asian
    'east_asian': {
        'qwen2.5-1.5b-instruct': {
            'provider': 'ollama',
            'origin': 'Alibaba (China)',
            'size': '1.5B',
            'languages': ['Chinese', 'English', 'multilingual'],
            'rationale': 'Primary non-Western anchor; strong instruction following',
            'quantization': None,
            'priority': 'high'
        },
        'qwen2.5-3b-instruct': {
            'provider': 'ollama',
            'origin': 'Alibaba (China)',
            'size': '3B',
            'languages': ['Chinese', 'English', 'multilingual'],
            'rationale': 'Primary non-Western anchor; strong instruction following',
            'quantization': None,
            'priority': 'medium'
        },
        'qwen2.5-7b-instruct': {
            'provider': 'ollama',
            'origin': 'Alibaba (China)',
            'size': '7B',
            'languages': ['Chinese', 'English', 'multilingual'],
            'rationale': 'Primary non-Western anchor; strong instruction following',
            'quantization': None,
            'priority': 'medium'
        },
    },

    # MENA / Arabic
    'mena_arabic': {
        'jais-13b-chat': {
            'provider': 'ollama',
            'origin': 'UAE (MBZUAI/G42)',
            'size': '13B',
            'languages': ['Arabic', 'English'],
            'rationale': 'Arabic-centric non-Western representation',
            'quantization': '4-bit',
            'priority': 'high',
            'status': 'check_availability'
        },
    },

    # Low-Resource / Global
    'global': {
        'phi3-mini': {
            'provider': 'ollama',
            'origin': 'Microsoft (Global)',
            'size': '3.8B',
            'languages': ['English', 'multilingual'],
            'rationale': 'Large-N CPU sweeps; robustness testing',
            'quantization': None,
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
    for category, category_models in MODELS.items():
        for model_name, config in category_models.items():
            if config.get('provider') == 'ollama':
                models.append(model_name)
    return models


def get_api_models():
    """Get list of models that require API access."""
    models = []
    for category, category_models in MODELS.items():
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


# Ollama model name mappings (exact names for pulling)
# NOTE: Ollama models don't use "-instruct" suffix
OLLAMA_MODEL_NAMES = {
    'gemma-2-2b': 'gemma2:2b',                    # Google Gemma 2 (2B params)
    'llama3.1-8b': 'llama3.1:8b',                 # Meta LLaMA 3.1 (8B params)
    'mistral-7b': 'mistral:7b',                   # Mistral AI (7B params)
    'qwen2.5-1.5b': 'qwen2.5:1.5b',               # Alibaba Qwen 2.5 (1.5B params)
    'qwen2.5-3b': 'qwen2.5:3b',                   # Alibaba Qwen 2.5 (3B params)
    'qwen2.5-7b': 'qwen2.5:7b',                   # Alibaba Qwen 2.5 (7B params)
    'phi3-mini': 'phi3:mini',                     # Microsoft Phi-3 Mini (3.8B params)
}

# Quick access to recommended models for each phase
RECOMMENDED_MODELS = {
    'testing': ['gemma2:2b', 'qwen2.5:1.5b'],           # Fast models for testing
    'baseline': ['gemma2:2b', 'qwen2.5:1.5b', 'phi3:mini'],  # Core baseline
    'full': ['gemma2:2b', 'llama3.1:8b', 'mistral:7b', 'qwen2.5:7b', 'phi3:mini'],  # Complete study
}