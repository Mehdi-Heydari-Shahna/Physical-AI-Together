"""A minimal proposal method for the cart (Section 10.3): it asks to
accelerate in every period. It carries no obligation for safety; the
governor alone keeps the cart within its constraints."""
from allovi import register


@register
class Eager:
    ROLE, NAME, CONTRACT = "proposal", "example.eager", "1.0"

    def propose(self, observation):
        return "accelerate"
