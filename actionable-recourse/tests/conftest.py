import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from recourse import *
from recourse.paths import *
from recourse.defaults import SUPPORTED_SOLVERS, _SOLVER_TYPE_CPX, _SOLVER_TYPE_PYTHON_MIP

@pytest.fixture(params = ['german', 'german-cat'])
def data(request):

    data_name = request.param
    data_file_name = test_dir / ('%s_processed.csv' % data_name.replace('-cat', ''))
    df = pd.read_csv(data_file_name)

    # process outcome name
    df_headers = df.columns.tolist()
    outcome_name = df_headers[0]
    df_headers.remove(outcome_name)

    # process names of variables to drop / categorical variables
    to_drop = [outcome_name]
    categorical_names = []
    if data_name == 'german':
        to_drop.extend(['Gender', 'OtherLoansAtStore', 'PurposeOfLoan'])
    elif data_name == 'german-cat':
        to_drop.extend(['Gender', 'OtherLoansAtStore'])
        categorical_names.extend(['PurposeOfLoan'])

    to_drop.extend(categorical_names)
    variable_names = [n for n in df_headers if n not in to_drop]

    # create data objects
    data = {
        'data_name': data_name,
        'outcome_name': outcome_name,
        'variable_names': variable_names,
        'categorical_names': categorical_names,
        'Y': df[outcome_name],
        'X': df[variable_names],
        'onehot_names': [],
        }

    if len(categorical_names) > 0:
        X_onehot = pd.get_dummies(df[categorical_names], prefix_sep = '_is_')
        data['X'] = data['X'].join(X_onehot)
        data['variable_names'] = data['X'].columns.tolist()
        data['onehot_names'] = X_onehot.columns.tolist()

    return data


@pytest.fixture(params = ['logreg'])
def classifier(request, data):
    if request.param == 'logreg':
        clf = LogisticRegression(max_iter = 1000, solver = 'lbfgs')
        clf.fit(data['X'], data['Y'])

    return clf


@pytest.fixture(params = ['logreg'])
def coefficients(request, classifier):
    if request.param == 'logreg':
        coef = classifier.coef_[0]
    return coef


@pytest.fixture
def scores(classifier, data):
    return pd.Series(classifier.predict_proba(data['X'])[:, 1])


@pytest.fixture(params=[1, 5, 8])
def denied_individual(request, scores, data, threshold):
    denied_individuals = scores.loc[lambda s: s <= threshold].index
    idx = denied_individuals[request.param]
    return data['X'].iloc[idx].values


@pytest.fixture(params=[.5])
def threshold(request, scores):
    return scores.quantile(request.param) ## or you can set a score-threshold, like .8


@pytest.fixture(params = ['actionable', 'immutable'])
def action_set(request, data):
    """Generate an action_set for German data."""
    # setup action_set

    action_set = ActionSet(X = data['X'])
    if request.param == 'immutable' and data['data_name'] == 'german':
        immutable_attributes = ['Age', 'Single', 'JobClassIsSkilled', 'ForeignWorker', 'OwnsHouse', 'RentsHouse']
        action_set[immutable_attributes].actionable = False
        action_set['CriticalAccountOrLoansElsewhere'].step_direction = -1
        action_set['CheckingAccountBalance_geq_0'].step_direction = 1

    return action_set


@pytest.fixture(params = ['neg'])
def features(request, data, classifier):
    yhat = classifier.predict(data['X'])
    if request.param == 'pos':
        idx = np.greater(yhat, 0)
    elif request.param == 'neg':
        idx = np.less_equal(yhat, 0)
    i = np.flatnonzero(idx)[0]
    x = np.array(data['X'].values[i, :])
    return x


@pytest.fixture(params = SUPPORTED_SOLVERS)
def recourse_builder(request, classifier, action_set):
    action_set.set_alignment(classifier)
    rb = RecourseBuilder(solver = request.param,
                         action_set = action_set,
                         clf = classifier)

    return rb


@pytest.fixture(params = SUPPORTED_SOLVERS)
def auditor(request, classifier, action_set):
    return RecourseAuditor(clf = classifier, action_set = action_set, solver= request.param)


@pytest.fixture(params = SUPPORTED_SOLVERS)
def flipset(request, classifier, action_set, denied_individual):
    print("request param")
    print(request.param)
    return Flipset(x = denied_individual, clf = classifier, action_set = action_set, solver= request.param)


@pytest.fixture
def recourse_builder_cpx(classifier, action_set):
    action_set.set_alignment(classifier)
    rb = RecourseBuilder(solver = _SOLVER_TYPE_CPX,
                         action_set = action_set,
                         clf = classifier)

    return rb


@pytest.fixture
def recourse_builder_python_mip(classifier, action_set):
    action_set.set_alignment(classifier)
    rb = RecourseBuilder(solver = _SOLVER_TYPE_PYTHON_MIP,
                         action_set = action_set,
                         clf = classifier)

    return rb


@pytest.fixture
def auditor_cpx(classifier, action_set):
    return RecourseAuditor(clf = classifier, action_set = action_set, solver= _SOLVER_TYPE_CPX)


@pytest.fixture
def auditor_python_mip(classifier, action_set):
    return RecourseAuditor(clf = classifier, action_set = action_set, solver= _SOLVER_TYPE_PYTHON_MIP)


@pytest.fixture
def flipset_cpx(classifier, action_set, denied_individual):
    return Flipset(x = denied_individual, clf = classifier, action_set = action_set, solver= _SOLVER_TYPE_CPX)


@pytest.fixture
def flipset_python_mip(classifier, action_set, denied_individual):
    return Flipset(x = denied_individual, clf = classifier, action_set = action_set, solver= _SOLVER_TYPE_PYTHON_MIP)
