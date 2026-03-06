"""
Geographic and Cultural Zone Metadata

Single source of truth for:
- ISO-3 country name display strings
- Inglehart-Welzel cultural zone assignments
- Zone color palette (matches the official WVS map scheme)
- Helper functions for zone lookup, quadrant labeling, and coordinate interpretation

Imported by both baseline_replication.py and visualize_baseline.py to avoid duplication.
"""

import pandas as pd

# ── Country display names (ISO-3 -> full name) ────────────────────────────────

COUNTRY_NAMES: dict[str, str] = {
    'ALB': 'Albania',        'DZA': 'Algeria',         'AND': 'Andorra',
    'ARG': 'Argentina',      'ARM': 'Armenia',          'AUS': 'Australia',
    'AUT': 'Austria',        'AZE': 'Azerbaijan',       'BGD': 'Bangladesh',
    'BLR': 'Belarus',        'BEL': 'Belgium',          'BOL': 'Bolivia',
    'BIH': 'Bosnia',         'BRA': 'Brazil',           'BGR': 'Bulgaria',
    'BFA': 'Burkina Faso',   'MMR': 'Myanmar',          'CAN': 'Canada',
    'CHL': 'Chile',          'CHN': 'China',            'COL': 'Colombia',
    'HRV': 'Croatia',        'CYP': 'Cyprus',           'CZE': 'Czechia',
    'ECU': 'Ecuador',        'EGY': 'Egypt',            'EST': 'Estonia',
    'ETH': 'Ethiopia',       'FIN': 'Finland',          'FRA': 'France',
    'GEO': 'Georgia',        'DEU': 'Germany',          'GHA': 'Ghana',
    'GRC': 'Greece',         'GTM': 'Guatemala',        'HTI': 'Haiti',
    'HKG': 'Hong Kong',      'HUN': 'Hungary',          'IND': 'India',
    'IDN': 'Indonesia',      'IRN': 'Iran',             'IRQ': 'Iraq',
    'IRL': 'Ireland',        'ITA': 'Italy',            'JPN': 'Japan',
    'JOR': 'Jordan',         'KAZ': 'Kazakhstan',       'KEN': 'Kenya',
    'KOR': 'South Korea',    'KWT': 'Kuwait',           'KGZ': 'Kyrgyzstan',
    'LBN': 'Lebanon',        'LBY': 'Libya',            'MAC': 'Macao',
    'MYS': 'Malaysia',       'MDV': 'Maldives',         'MLI': 'Mali',
    'MEX': 'Mexico',         'MDA': 'Moldova',          'MNG': 'Mongolia',
    'MAR': 'Morocco',        'NLD': 'Netherlands',      'NZL': 'New Zealand',
    'NIC': 'Nicaragua',      'NGA': 'Nigeria',          'NIR': 'N. Ireland',
    'NOR': 'Norway',         'PAK': 'Pakistan',         'PSE': 'Palestine',
    'PER': 'Peru',           'PHL': 'Philippines',      'POL': 'Poland',
    'PRI': 'Puerto Rico',    'QAT': 'Qatar',            'ROU': 'Romania',
    'RUS': 'Russia',         'RWA': 'Rwanda',           'SRB': 'Serbia',
    'SGP': 'Singapore',      'SVK': 'Slovakia',         'SVN': 'Slovenia',
    'ZAF': 'South Africa',   'ESP': 'Spain',            'SWE': 'Sweden',
    'CHE': 'Switzerland',    'TJK': 'Tajikistan',       'TWN': 'Taiwan',
    'THA': 'Thailand',       'TTO': 'Trinidad',         'TUN': 'Tunisia',
    'TUR': 'Turkey',         'UKR': 'Ukraine',          'GBR': 'UK',
    'USA': 'USA',            'URY': 'Uruguay',          'UZB': 'Uzbekistan',
    'VEN': 'Venezuela',      'VNM': 'Vietnam',          'YEM': 'Yemen',
    'ZMB': 'Zambia',         'ZWE': 'Zimbabwe',
}

# ── Inglehart-Welzel cultural zones ──────────────────────────────────────────

CULTURAL_ZONES: dict[str, list[str]] = {
    'Protestant Europe': [
        'CHE', 'DEU', 'FIN', 'GBR', 'NLD', 'NOR', 'SWE', 'EST', 'NIR',
    ],
    'Catholic Europe': [
        'AND', 'AUT', 'BEL', 'CYP', 'CZE', 'FRA', 'GRC', 'HRV', 'HUN',
        'IRL', 'ITA', 'POL', 'PRT', 'SVK', 'SVN', 'ESP',
    ],
    'English-speaking': [
        'AUS', 'CAN', 'NZL', 'PRI', 'USA',
    ],
    'Orthodox': [
        'ARM', 'AZE', 'BLR', 'BGR', 'GEO', 'MDA', 'ROU', 'RUS', 'SRB', 'UKR',
    ],
    'East Asia': [
        'CHN', 'HKG', 'JPN', 'KOR', 'MAC', 'MNG', 'SGP', 'TWN', 'VNM',
    ],
    'Latin America': [
        'ARG', 'BOL', 'BRA', 'CHL', 'COL', 'ECU', 'GTM', 'HTI',
        'MEX', 'NIC', 'PER', 'TTO', 'URY', 'VEN',
    ],
    'South/SE Asia': [
        'BGD', 'IDN', 'IND', 'MDV', 'MMR', 'MYS', 'PAK', 'PHL', 'THA',
    ],
    'Africa': [
        'BFA', 'ETH', 'GHA', 'KEN', 'MLI', 'NGA', 'RWA', 'ZAF', 'ZMB', 'ZWE',
    ],
    'MENA': [
        'DZA', 'EGY', 'IRN', 'IRQ', 'JOR', 'KWT', 'LBN', 'LBY',
        'MAR', 'PSE', 'QAT', 'TUN', 'TUR', 'YEM',
    ],
    'Central Asia': [
        'KAZ', 'KGZ', 'TJK', 'UZB',
    ],
}

# Flat reverse lookup: ISO-3 -> zone name
ISO3_TO_ZONE: dict[str, str] = {
    iso3: zone for zone, members in CULTURAL_ZONES.items() for iso3 in members
}

# Zone colors matching the official Inglehart-Welzel palette
ZONE_COLORS: dict[str, str] = {
    'Protestant Europe': '#4477AA',
    'Catholic Europe':   '#AACCEE',
    'English-speaking':  '#44AA77',
    'Orthodox':          '#AA3377',
    'East Asia':         '#CC3311',
    'Latin America':     '#EE7733',
    'South/SE Asia':     '#CCBB44',
    'Africa':            '#885522',
    'MENA':              '#BBBBBB',
    'Central Asia':      '#CC99BB',
}

# ── Helper functions ──────────────────────────────────────────────────────────

def load_iso3_lookup(ivs_path) -> dict[int, str]:
    """Build numeric IVS country code (S003) -> ISO-3 lookup from processed IVS data."""
    df = pd.read_csv(ivs_path, usecols=['S003', 'COUNTRY_ALPHA'], low_memory=False)
    return df.dropna().drop_duplicates('S003').set_index('S003')['COUNTRY_ALPHA'].to_dict()


def get_zone(iso3: str) -> str:
    """Return the cultural zone for an ISO-3 code, or 'Other' if unknown."""
    return ISO3_TO_ZONE.get(iso3, 'Other')


def quadrant_label(x: float, y: float) -> str:
    """Return a human-readable quadrant label for a (x, y) cultural map position."""
    h = 'Self-Expression' if x > 0 else 'Survival'
    v = 'Secular' if y > 0 else 'Traditional'
    return f'{h} + {v}'


def typical_countries(x: float, y: float) -> str:
    """Return a short geographic description for a (x, y) position on the cultural map."""
    if x > 2 and y > 0:        return 'Nordic Europe, East Asia'
    if x > 2 and y < 0:        return 'Nordic/Western Europe, English-speaking'
    if 0 < x <= 2 and y > 0:   return 'Eastern Europe, East Asia'
    if 0 < x <= 2 and y < 0:   return 'Southern Europe, Latin America, USA'
    if x < 0 and y > 0:        return 'MENA, Central Asia, Orthodox Europe'
    if x < 0 and y < 0:        return 'Sub-Saharan Africa, South Asia, Latin America'
    return 'Near origin (central cluster)'
