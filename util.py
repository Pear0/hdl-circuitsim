import re

_signal_name_re = re.compile(r'([a-zA-Z_0-9]+)(?:\((\d+)(?:-(\d+))?\))?', re.IGNORECASE)


def parse_signal(signal):
    sig_match = _signal_name_re.match(signal)
    if not sig_match:
        raise ValueError('Could not parse: ' + signal)

    if sig_match.group(2):
        end = int(sig_match.group(2))
        start = end
        if sig_match.group(3):
            start = int(sig_match.group(3))

        return sig_match.group(1), end, start
    else:
        return sig_match.group(1)
