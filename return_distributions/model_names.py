"""Canonical names shared by fitting and command-line model selectors."""
MODEL_ALIASES = {name: name+'-skewed' for name in
                 ('nig', 'gh', 'hyperbolic', 'variance-gamma')}
MODEL_ALIASES.update(meixner='meixner-skewed', **{'nef-ghs':'meixner-skewed'})
MODEL_ALIASES['egb2'] = 'egb2-skewed'
MODEL_ALIASES['nts'] = 'nts-skewed'
MODEL_ALIASES['laplace-skewed'] = 'laplace_asymmetric'
MODEL_ALIASES['generalized-t-symmetric'] = 'generalized-t'
MODEL_ALIASES.update({'johnson-su': 'johnsonsu', 'johnson-su-skewed': 'johnsonsu',
                      'crystal-ball': 'crystalball'})


def canonical_model(name):
    return MODEL_ALIASES.get(name, name)


def unique_models(names):
    return list(dict.fromkeys(canonical_model(name) for name in names))


def display_model_name(name):
    """Preferred help spelling only; accepted names and saved identifiers are unchanged."""
    preferred = {
        'azzalini': 'azzalini-skew-t',
        'azzalini_skew_t': 'azzalini-skew-t',
        'fernandez-steel': 'fs-skew-t',
        'fs_skew_t': 'fs-skew-t',
        'generalized-t-symmetric': 'generalized-t',
        'laplace_asymmetric': 'laplace-skewed',
        'nef-ghs': 'meixner',
        'johnsonsu': 'johnson-su',
        'crystalball': 'crystal-ball',
    }
    if name in preferred:
        return preferred[name]
    if name.endswith('-skewed'):
        base = name[:-len('-skewed')]
        if canonical_model(base) == canonical_model(name):
            return base
    return name


def display_model_label(label):
    """Short literature names for console labels; never change stored identifiers."""
    if not isinstance(label, str):
        return label
    head, separator, tail = label.partition(' ')
    preferred = {name+'-skewed': name for name in
                 ('nig', 'gh', 'hyperbolic', 'variance-gamma', 'meixner', 'egb2')}
    return preferred.get(head, head) + separator + tail
