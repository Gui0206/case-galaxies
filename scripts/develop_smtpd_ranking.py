"""Test a direct ranking objective on pre-2024 development data only."""

import json

from galaxies.smtpd_ranking import develop

print(json.dumps(develop(), indent=2))
