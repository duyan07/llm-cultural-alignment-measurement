"""
Response Parser

Extracts structured answers from LLM text responses.
Handles different question types from the IVS questions.
"""

import re
from typing import Optional, Union, List, Any


class ResponseParser:
    """
    Extracts structured answers from LLM text responses, matching one of the
    four IVS question formats: numeric, categorical, multi-numeric (Y002),
    or multi-categorical (Y003).
    """

    @staticmethod
    def parse_numeric(response: str, scale: tuple = (1, 10)) -> Optional[float]:
        """Extract the first in-range number from response, or None if invalid."""
        numbers = re.findall(r'\b\d+\.?\d*\b', response)

        if not numbers:
            return None

        try:
            value = float(numbers[0])
            min_val, max_val = scale

            if min_val <= value <= max_val:
                return value
        except ValueError:
            pass

        return None

    @staticmethod
    def parse_categorical(response: str, options: List[str]) -> Optional[str]:
        """Extract a single option letter (e.g. 'A') from response, or None if none matched."""
        response_upper = response.upper()

        for option in options:
            option_upper = option.upper()
            if re.search(rf'\b{option_upper}\b', response_upper):
                return option

        return None

    @staticmethod
    def parse_multi_numeric(response: str, count: int = 2,
                          options: List[int] = [1, 2, 3, 4]) -> Optional[List[int]]:
        """Extract `count` distinct numbers from response (e.g. Y002: choose 2 from 4)."""
        numbers = re.findall(r'\b\d+\b', response)

        if not numbers:
            return None

        selected = []
        for num_str in numbers:
            try:
                num = int(num_str)
                if num in options and num not in selected:
                    selected.append(num)
                    if len(selected) == count:
                        break
            except ValueError:
                continue

        if len(selected) == count:
            return selected

        return None

    @staticmethod
    def parse_multi_categorical(response: str, options: List[str],
                               max_count: int = 5) -> Optional[List[str]]:
        """Extract up to `max_count` matching options from response (e.g. Y003: choose up to 5)."""
        response_lower = response.lower()
        selected = []

        for option in options:
            option_lower = option.lower()
            if option_lower in response_lower:
                selected.append(option)
                if len(selected) >= max_count:
                    break

        return selected if selected else None

    @staticmethod
    def parse_by_type(response: str, question_info: dict) -> Optional[Any]:
        """Dispatch to the parser matching question_info['response_type']."""
        response_type = question_info['response_type']

        if response_type == 'numeric':
            scale = question_info.get('scale', (1, 10))
            return ResponseParser.parse_numeric(response, scale)

        elif response_type == 'categorical':
            options = question_info['options']
            return ResponseParser.parse_categorical(response, options)

        elif response_type == 'multi_numeric':
            count = question_info.get('count', 2)
            options = question_info.get('options', [1, 2, 3, 4])
            return ResponseParser.parse_multi_numeric(response, count, options)

        elif response_type == 'multi_categorical':
            options = question_info['options']
            max_count = question_info.get('max_count', 5)
            return ResponseParser.parse_multi_categorical(response, options, max_count)

        else:
            return None

    @staticmethod
    def validate_response(response: str, question_info: dict) -> dict:
        """Parse a response and return it alongside validity and question metadata."""
        parsed = ResponseParser.parse_by_type(response, question_info)

        return {
            'raw_response': response,
            'parsed_value': parsed,
            'is_valid': parsed is not None,
            'response_type': question_info['response_type'],
            'question_id': question_info['ivs_variable']
        }


    @staticmethod
    def to_ivs_numeric(parsed_value: Any, question_info: dict) -> Optional[float]:
        """
        Convert a parsed response to its IVS numeric equivalent for PCA input.

        The PCA pipeline expects all inputs to be numeric, so categorical and
        multi-choice questions need conversion:
          - categorical: map letter position to ordinal (A=1, B=2, C=3)
          - Y002: derive post-materialist index (1-4) from two chosen goal numbers
          - Y003: binary autonomy index (1 if 'Independence' selected, else 0)

        Returns None if conversion fails.
        """
        if parsed_value is None:
            return None

        response_type = question_info['response_type']
        question_id = question_info['ivs_variable']

        if response_type == 'numeric':
            return float(parsed_value)

        elif response_type == 'categorical':
            # Map letter to its ordinal position in the options list (A=1, B=2, C=3)
            options = question_info['options']
            if parsed_value in options:
                return float(options.index(parsed_value) + 1)
            return None

        elif response_type == 'multi_numeric':
            # Y002: classify two chosen goals into the 4-point post-materialist index
            # Goals 2 (more say) and 4 (freedom of speech) are post-materialist
            # Goals 1 (maintaining order) and 3 (fighting prices) are materialist
            if question_id == 'Y002' and isinstance(parsed_value, list) and len(parsed_value) == 2:
                postmat = {2, 4}
                n_postmat = len(set(parsed_value) & postmat)
                if n_postmat == 0:
                    return 1.0  # Both materialist
                elif n_postmat == 2:
                    return 4.0  # Both post-materialist
                else:
                    # One of each: 2 if first choice is materialist, 3 if first is post-materialist
                    return 3.0 if parsed_value[0] in postmat else 2.0
            return None

        elif response_type == 'multi_categorical':
            # Y003: binary indicator — 1 if 'Independence' was selected, 0 otherwise
            if question_id == 'Y003' and isinstance(parsed_value, list):
                return 1.0 if 'Independence' in parsed_value else 0.0
            return None

        return None


# Convenience functions for common cases
def extract_number(response: str, min_val: int = 1, max_val: int = 10) -> Optional[float]:
    """Extract a single number from response."""
    return ResponseParser.parse_numeric(response, scale=(min_val, max_val))


def extract_letter(response: str, valid_letters: List[str] = ['A', 'B', 'C']) -> Optional[str]:
    """Extract a single letter choice from response."""
    return ResponseParser.parse_categorical(response, valid_letters)