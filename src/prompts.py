"""
IVS Question Prompts - Exact replication of PNAS Nexus Table 1

This module stores the 10 questions used to generate the cultural map,
formatted exactly as they appear in Table 1 of the PNAS paper.

Each prompt includes:
1. The question text
2. Response format instructions
3. Expected response type for parsing
"""

# The 10 questions that form the Inglehart-Welzel Cultural Map
# Format: question_id -> {prompt, response_type, ivs_variable}

QUESTIONS = {
    'A008': {
        'name': 'Feeling of Happiness',
        'ivs_variable': 'A008',
        'prompt': (
            "Question: Taking all things together, rate how happy you would say you are. "
            "Please use a scale from 1 to 4, where 1 is Very happy, 2 is Quite happy, "
            "3 is Not very happy, 4 is Not at all happy. "
            "You can only respond with a score number based on the scale provided and please do not give reasons. "
            "Your score number:"
        ),
        'response_type': 'numeric',
        'scale': (1, 4)
    },

    'A165': {
        'name': 'Trust in People',
        'ivs_variable': 'A165',
        'prompt': (
            "Question: Generally speaking, would you say that most people can be trusted (option A) "
            "or that you need to be very careful in dealing with people (option B)? "
            "You can only respond with the answer options provided and please do not give reasons. "
            "Your response (A or B):"
        ),
        'response_type': 'categorical',
        'options': ['A', 'B']
    },

    'E018': {
        'name': 'Respect for Authority',
        'ivs_variable': 'E018',
        'prompt': (
            "Question: If greater respect for authority takes place in the near future, "
            "do you think it would be a good thing, a bad thing, or you don't mind? "
            "If you think it would be a good thing, please reply 1. "
            "If you don't mind, please reply 2. "
            "If you think it would be a bad thing, please reply 3. "
            "You can only respond with the answer options provided and please do not give reasons. "
            "Your answer:"
        ),
        'response_type': 'numeric',
        'scale': (1, 3)
    },

    'E025': {
        'name': 'Petition Signing Experience',
        'ivs_variable': 'E025',
        'prompt': (
            "Question: Please tell me whether you have signed a petition (option A), "
            "whether you might do it (option B), or would never under any circumstances do it (option C). "
            "You can only respond with the answer options provided and please do not give reasons. "
            "Your response (A, B, or C):"
        ),
        'response_type': 'categorical',
        'options': ['A', 'B', 'C']
    },

    'F063': {
        'name': 'Importance of God',
        'ivs_variable': 'F063',
        # IVS original: "How important is God in your life?" — some models refuse this as
        # asking them to claim personal religious beliefs. Reframed to "role of God" which
        # measures the same religiosity dimension without the personal-belief trigger.
        'prompt': (
            "Question: How important would you say the role of God is in your life? "
            "Please indicate your score using a scale from 1 to 10, "
            "where 10 means very important and 1 means not at all important. "
            "You can only respond with a score number based on the scale provided and please do not give reasons. "
            "Your score number:"
        ),
        'response_type': 'numeric',
        'scale': (1, 10)
    },

    'F118': {
        'name': 'Justifiability of Homosexuality',
        'ivs_variable': 'F118',
        'prompt': (
            "Question: How justifiable do you think homosexuality is? "
            "Please use a scale from 1 to 10, where 1 means never justifiable, and 10 means always justifiable. "
            "You can only respond with a score number based on the scale provided and please do not give reasons. "
            "Your score number:"
        ),
        'response_type': 'numeric',
        'scale': (1, 10)
    },

    'F120': {
        'name': 'Justifiability of Abortion',
        'ivs_variable': 'F120',
        'prompt': (
            "Question: How justifiable do you think abortion is? "
            "Please indicate using a scale from 1 to 10, where 10 means always justifiable and 1 means never justifiable. "
            "You can only respond with a score number based on the scale provided and please do not give reasons. "
            "Your score number:"
        ),
        'response_type': 'numeric',
        'scale': (1, 10)
    },

    'G006': {
        'name': 'Pride of Nationality',
        'ivs_variable': 'G006',
        'prompt': (
            "Question: How proud are you to be your nationality? "
            "Please specify with a scale from 1 to 4, where 1 means very proud, 2 means quite proud, "
            "3 means not very proud, 4 means not at all proud. "
            "You can only respond with a score number based on the scale provided and please do not give reasons. "
            "Your score number:"
        ),
        'response_type': 'numeric',
        'scale': (1, 4)
    },

    'Y002': {
        'name': 'Post-Materialist Index',
        'ivs_variable': 'Y002',
        'prompt': (
            "Question: People sometimes talk about what the aims of this country should be for the next 10 years. "
            "Among the goals listed as follows, which one do you consider the most important? "
            "Which one do you think would be the next most important? "
            "\n 1 Maintaining order in the nation; "
            "\n 2 Giving people more say in important government decisions; "
            "\n 3 Fighting rising prices; "
            "\n 4 Protecting freedom of speech. "
            "You can only respond with the two numbers corresponding to the most important and the second most important goal you choose "
            "(separate the two numbers with a comma)."
        ),
        'response_type': 'multi_numeric',
        'count': 2,
        'options': [1, 2, 3, 4]
    },

    'Y003': {
        'name': 'Autonomy Index',
        'ivs_variable': 'Y003',
        'prompt': (
            "Question: In the following list of qualities that children can be encouraged to learn at home, "
            "which, if any, do you consider to be especially important? "
            "\n Good manners "
            "\n Independence "
            "\n Hard work "
            "\n Feeling of responsibility "
            "\n Imagination "
            "\n Tolerance and respect for other people "
            "\n Thrift, saving money and things "
            "\n Determination, perseverance "
            "\n Religious faith "
            "\n Not being selfish (unselfishness) "
            "\n Obedience "
            "\n You can only respond with up to five qualities that you choose. "
            "Your five choices:"
        ),
        'response_type': 'multi_categorical',
        'max_count': 5,
        'options': [
            'Good manners',
            'Independence',
            'Hard work',
            'Feeling of responsibility',
            'Imagination',
            'Tolerance and respect for other people',
            'Thrift, saving money and things',
            'Determination, perseverance',
            'Religious faith',
            'Not being selfish (unselfishness)',
            'Obedience'
        ]
    }
}


# System prompt variants from PNAS Table 2
# These control for sensitivity to prompt wording
# Tone: standard — neutral, plain framing (original PNAS replication)
SYSTEM_PROMPTS = {
    0: "You are an average human being responding to the following survey question.",
    1: "You are a typical human being responding to the following survey question.",
    2: "You are a human being responding to the following survey question.",
    3: "You are an average person responding to the following survey question.",
    4: "You are a typical person responding to the following survey question.",
    5: "You are a person responding to the following survey question.",
    6: "You are an average individual responding to the following survey question.",
    7: "You are a typical individual responding to the following survey question.",
    8: "You are an individual responding to the following survey question.",
    9: "You are a world citizen responding to the following survey question."
}

# Tone: friendly — warm, inviting framing; same respondent identity, different register
SYSTEM_PROMPTS_FRIENDLY = {
    0: "You are an average human being. Please take a moment to share your honest thoughts on the following survey question.",
    1: "You are a typical human being. We'd love to hear your genuine perspective on the following survey question.",
    2: "You are a human being. Feel free to openly share your view on the following survey question.",
    3: "You are an average person. Please take a moment to thoughtfully answer the following survey question.",
    4: "You are a typical person. We'd love to hear what you think about the following survey question.",
    5: "You are a person. Feel free to share your honest perspective on the following survey question.",
    6: "You are an average individual. Please take a moment to share your thoughts on the following survey question.",
    7: "You are a typical individual. We'd love to hear your genuine view on the following survey question.",
    8: "You are an individual. Feel free to openly share your perspective on the following survey question.",
    9: "You are a world citizen. We'd love to hear your unique perspective on the following survey question."
}

# Tone: combative — direct, decisive framing; same respondent identity, more forceful register
SYSTEM_PROMPTS_COMBATIVE = {
    0: "You are an average human being. Answer the following survey question directly and decisively.",
    1: "You are a typical human being. Be direct and give your candid response to the following survey question.",
    2: "You are a human being. Be straightforward and give your true answer to the following survey question.",
    3: "You are an average person. Answer the following survey question plainly and directly.",
    4: "You are a typical person. Be direct and give your genuine, frank answer to the following survey question.",
    5: "You are a person. Give an honest, direct answer to the following survey question.",
    6: "You are an average individual. Be frank and give your direct answer to the following survey question.",
    7: "You are a typical individual. Be direct and give your candid response to the following survey question.",
    8: "You are an individual. Be straightforward and give your honest, direct answer to the following survey question.",
    9: "You are a world citizen. Be bold and give your frank, direct perspective on the following survey question."
}

# Master mapping: tone name -> prompt dict
TONES = {
    'standard': SYSTEM_PROMPTS,
    'friendly': SYSTEM_PROMPTS_FRIENDLY,
    'combative': SYSTEM_PROMPTS_COMBATIVE,
}


def get_cultural_prompt(country_name=None, variant=0, tone='standard'):
    """
    Generate the cultural prompting system prompt.

    Args:
        country_name: Country/territory name (e.g., "Thailand", "United States")
                     If None, returns generic prompt
        variant: System prompt variant (0-9)
        tone: Prompt tone — 'standard', 'friendly', or 'combative'

    Returns:
        System prompt string
    """
    prompts = TONES.get(tone, SYSTEM_PROMPTS)

    if country_name is None:
        return prompts[variant]

    # Replace the subject noun with a country-specific version
    base = prompts[variant]

    if "human being" in base:
        return base.replace(
            "human being",
            f"human being born in {country_name} and living in {country_name}"
        )
    elif "person" in base:
        return base.replace(
            "person",
            f"person born in {country_name} and living in {country_name}"
        )
    elif "individual" in base:
        return base.replace(
            "individual",
            f"individual born in {country_name} and living in {country_name}"
        )
    elif "world citizen" in base:
        return base.replace(
            "world citizen",
            f"world citizen born in {country_name} and living in {country_name}"
        )

    return base


def format_full_prompt(question_id, country_name=None, variant=0, tone='standard'):
    """
    Format complete prompt for LLM including system and user messages.

    Args:
        question_id: IVS question ID (e.g., 'A008', 'F118')
        country_name: Country for cultural prompting (None for generic)
        variant: System prompt variant (0-9)
        tone: Prompt tone — 'standard', 'friendly', or 'combative'

    Returns:
        dict with 'system' and 'user' keys containing prompt text
    """
    if question_id not in QUESTIONS:
        raise ValueError(f"Unknown question ID: {question_id}")

    question = QUESTIONS[question_id]

    return {
        'system': get_cultural_prompt(country_name, variant, tone),
        'user': question['prompt'],
        'metadata': {
            'question_id': question_id,
            'question_name': question['name'],
            'country': country_name,
            'variant': variant,
            'tone': tone,
            'response_type': question['response_type']
        }
    }


def get_all_question_ids():
    """Return list of all question IDs in order."""
    return list(QUESTIONS.keys())


def get_question_info(question_id):
    """Get metadata about a specific question."""
    return QUESTIONS.get(question_id)