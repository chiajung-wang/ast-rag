import os

# provider.api_key() raises when OPENROUTER_API_KEY is unset, which is the
# right behaviour for the app: fail loudly at startup rather than at the first
# request. Tests construct clients (always patched, never called) so they need
# the variable present but not valid.
#
# Set a dummy rather than a real key. CI explicitly blanks the variable to
# prove the suite needs no credentials, and this keeps that true: no test
# reaches the network, and a real key here would hide it if one did.
# Not setdefault: CI sets the variable to an empty string, which setdefault
# treats as already present and leaves alone.
if not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = "test-key-not-used"
