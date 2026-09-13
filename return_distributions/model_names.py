"""Canonical names shared by fitting and command-line model selectors."""
MODEL_ALIASES = {name: name+'-skewed' for name in
                 ('nig', 'gh', 'hyperbolic', 'variance-gamma')}
MODEL_ALIASES.update(meixner='meixner-skewed', **{'nef-ghs':'meixner-skewed'})
MODEL_ALIASES['egb2'] = 'egb2-skewed'
MODEL_ALIASES['nts'] = 'nts-skewed'


def canonical_model(name):
    return MODEL_ALIASES.get(name, name)


def unique_models(names):
    return list(dict.fromkeys(canonical_model(name) for name in names))
