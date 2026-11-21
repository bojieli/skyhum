"""Exit codes and the exceptions that map to them."""

OK = 0            # success; for verify: SUPPORTED
FAILED = 1        # selftest failed or an internal error
USAGE = 2         # bad arguments, or a feature not implemented in this build
MISSING_DATA = 3  # ADS-B, weights or another input the command needs is not available
REFUSED = 4       # the find safety gate refused the request
UNCORROBORATED = 10
INCONCLUSIVE = 11

VERDICT_EXIT = {"SUPPORTED": OK, "UNCORROBORATED": UNCORROBORATED, "INCONCLUSIVE": INCONCLUSIVE}


class SkyhumError(Exception):
    exit_code = FAILED


class UsageError(SkyhumError):
    exit_code = USAGE


class NotImplementedInBuild(SkyhumError):
    """A documented feature that this build does not have yet."""

    exit_code = USAGE

    def __init__(self, what, why=""):
        msg = f"{what} is not yet implemented in this build"
        if why:
            msg += f": {why}"
        super().__init__(msg)


class MissingData(SkyhumError):
    exit_code = MISSING_DATA


class Refused(SkyhumError):
    exit_code = REFUSED
