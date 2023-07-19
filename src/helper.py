from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def get_base_models(dataset, seed=None):
    if dataset == 'adult':
        return {
            'lr': LogisticRegression(solver='liblinear', fit_intercept=True, random_state=seed),
            'rf': RandomForestClassifier(min_samples_leaf=5, max_depth=5, random_state=seed),
            'avd_debaising': MLPClassifier(max_iter=500, random_state=seed),
            'gbm': GradientBoostingClassifier(n_estimators=5, learning_rate=0.1, max_depth=5, random_state=seed)
        }

    if dataset == 'compas_race':
        return {
            'lr': LogisticRegression(solver='liblinear', fit_intercept=True, random_state=seed),
            'rf': RandomForestClassifier(min_samples_leaf=5, max_depth=5, random_state=seed),
            'avd_debaising': MLPClassifier(max_iter=500, random_state=seed),
            'gbm': GradientBoostingClassifier(learning_rate=0.1, max_depth=5, random_state=seed)
        }

    return {
        'lr': LogisticRegression(solver='liblinear', fit_intercept=True, random_state=seed),
        'rf': RandomForestClassifier(min_samples_leaf=5, max_depth=15, random_state=seed),
        'avd_debaising': MLPClassifier(max_iter=500, random_state=seed),
        'gbm': GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=seed)
    }
