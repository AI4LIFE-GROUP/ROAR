import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.autograd import grad
import datetime
#from recourse.builder import RecourseBuilder
#from recourse.builder import ActionSet
from scipy.optimize import linprog
from recourse_utils import *
from tqdm import tqdm


def counterfactual_recourse(torch_model, x, feature_costs=None, y_target=1.0, n_iter=100, tmax_min=5):
    # returns x'
    torch.manual_seed(0)

    if feature_costs is not None:
        feature_costs = torch.from_numpy(feature_costs).float()

    x = torch.from_numpy(x).float()
    y_target = torch.tensor(y_target).float()
    lamb = torch.tensor(0.01).float()

    x_new = Variable(x.clone(), requires_grad=True)
    f_x_new = torch_model(x_new)
    optimizer = optim.Adam([x_new], amsgrad=True)

    loss_fn = torch.nn.MSELoss()
    t0 = datetime.datetime.now()
    tmax = datetime.timedelta(minutes=tmax_min)

    while f_x_new < 0.51:
        it = 0
        while f_x_new < 0.51 and it < n_iter:
            optimizer.zero_grad()

            f_x_new = torch_model(x_new)

            if feature_costs is None:
                cost = torch.dist(x_new, x, 1)
            else:
                cost = torch.norm(feature_costs * (x_new - x), 1)

            loss = lamb * loss_fn(f_x_new, y_target) + cost
            loss.backward()
            optimizer.step()
            it += 1

        if datetime.datetime.now() - t0 > tmax:
            print("timeout")
            break

        if lamb > 1:
            lamb += (lamb / 10)
        else:
            lamb += 0.1

    return x_new.detach().numpy()


'''
def actionable_recourse(x,X_train, coefficients, intercept, cost_type, feature_costs=None):
	action_set = ActionSet(X=X_train)
	rb = RecourseBuilder(
		  optimizer="cplex",
		  coefficients=coefficients,
		  intercept=intercept,
		  action_set=action_set,
		  x=x,mip_cost_type=cost_type,
		  pwfeature_costs=feature_costs
	)
	r = rb.fit()
	if r['feasible']:
		ur = x+r["actions"]
	else:
		ur = x
		"Failed to find recourse"
	return ur 
'''


class RobustRecourse():
    def __init__(self, W=None, W0=None, y_target=1,
                 delta_max=0.1, feature_costs=None,
                 pW=None, pW0=None):
        self.set_W(W)
        self.set_W0(W0)

        self.set_pW(pW)
        self.set_pW0(pW0)

        self.y_target = torch.tensor(y_target).float()
        self.delta_max = delta_max
        self.feature_costs = feature_costs
        if self.feature_costs is not None:
            self.feature_costs = torch.from_numpy(feature_costs).float()

    def set_W(self, W):
        self.W = W
        if W is not None:
            self.W = torch.from_numpy(W).float()

    def set_W0(self, W0):
        self.W0 = W0
        if W0 is not None:
            self.W0 = torch.from_numpy(W0).float()

    def set_pW(self, pW):
        self.pW = pW
        if pW is not None:
            self.pW = torch.from_numpy(pW).float()

    def set_pW0(self, pW0):
        self.pW0 = pW0
        if pW0 is not None:
            self.pW0 = torch.from_numpy(pW0).float()

    def l1_cost(self, x_new, x):
        cost = torch.dist(x_new, x, 1)
        return cost

    def pfc_cost(self, x_new, x):
        cost = torch.norm(self.feature_costs * (x_new - x), 1)
        return cost

    def calc_delta_opt(self, recourse):
        """
		calculate the optimal delta using linear program
		:returns: torch tensor with optimal delta value
		"""
        W = torch.cat((self.W, self.W0), 0)  # Add intercept to weights
        recourse = torch.cat((recourse, torch.ones(1)), 0)  # Add 1 to the feature vector for intercept

        loss_fn = torch.nn.BCELoss()

        A_eq = np.empty((0, len(W)), float)

        b_eq = np.array([])

        W.requires_grad = True
        f_x_new = torch.nn.Sigmoid()(torch.matmul(W, recourse))
        w_loss = loss_fn(f_x_new, self.y_target)
        gradient_w_loss = grad(w_loss, W)[0]

        c = list(np.array(gradient_w_loss) * np.array([-1] * len(gradient_w_loss)))
        bound = (-self.delta_max, self.delta_max)
        bounds = [bound] * len(gradient_w_loss)

        res = linprog(c, bounds=bounds, A_eq=A_eq, b_eq=b_eq, method='simplex')
        delta_opt = res.x  # the delta value that maximizes the function
        delta_W, delta_W0 = np.array(delta_opt[:-1]), np.array([delta_opt[-1]])
        return delta_W, delta_W0

    def get_recourse(self, x, lamb=0.1):
        torch.manual_seed(0)

        # returns x'
        x = torch.from_numpy(x).float()
        lamb = torch.tensor(lamb).float()

        x_new = Variable(x.clone(), requires_grad=True)
        optimizer = optim.Adam([x_new])

        loss_fn = torch.nn.BCELoss()

        # Placeholders
        loss = torch.tensor(1)
        loss_diff = 1
        f_x_new = 0

        while loss_diff > 1e-4:
            loss_prev = loss.clone().detach()

            delta_W, delta_W0 = self.calc_delta_opt(x_new)
            delta_W, delta_W0 = torch.from_numpy(delta_W).float(), torch.from_numpy(delta_W0).float()

            optimizer.zero_grad()
            if self.pW is not None:
                dec_fn = torch.matmul(self.W + delta_W, x_new) + self.W0
                f_x_new = torch.nn.Sigmoid()(torch.matmul(self.pW, dec_fn.unsqueeze(0)) + self.pW0)[0]
            else:
                f_x_new = torch.nn.Sigmoid()(torch.matmul(self.W + delta_W, x_new) + self.W0 + delta_W0)[0]

            if self.feature_costs is not None:
                cost = self.pfc_cost(x_new, x)
            else:
                cost = self.l1_cost(x_new, x)

            loss = loss_fn(f_x_new, self.y_target) + lamb * cost
            loss.backward()
            optimizer.step()

            loss_diff = torch.dist(loss_prev, loss, 2)
        return x_new.detach().numpy(), np.concatenate((delta_W.detach().numpy(), delta_W0.detach().numpy()))

    # Heuristic for picking hyperparam lambda
    def choose_lambda(self, recourse_needed_X, predict_fn, X_train=None, predict_proba_fn=None):
        lambdas = np.arange(0.1, 1.1, 0.1)

        v_old = 0
        for i, lamb in enumerate(lambdas):
            print("Testing lambda:%f" % lamb)
            recourses = []
            for xi, x in tqdm(enumerate(recourse_needed_X)):

                # Call lime if nonlinear
                if self.W0 is None and self.W is None:
                    # set seed for lime
                    np.random.seed(xi)
                    coefficients, intercept = lime_explanation(predict_proba_fn,
                                                               X_train, x)
                    coefficients, intercept = np.round_(coefficients, 4), np.round_(intercept, 4)
                    self.set_W(coefficients)
                    self.set_W0(intercept)

                    r, _ = self.get_recourse(x, lamb)

                    self.set_W(None)
                    self.set_W0(None)
                else:
                    r, _ = self.get_recourse(x, lamb)
                recourses.append(r)

            v = recourse_validity(predict_fn, recourses, target=self.y_target.numpy())
            if v >= v_old:
                v_old = v
            else:
                li = max(0, i - 1)
                return lambdas[li]

        return lamb

    def choose_delta(self, recourse_needed_X, predict_fn, X_train=None,
                     predict_proba_fn=None, lamb=0.1):
        deltas = [0.1, 0.25, 0.5, 0.75]

        v_old = 0
        for i, d in enumerate(deltas):
            print("Testing delta:%f" % d)
            recourses = []
            for xi, x in tqdm(enumerate(recourse_needed_X)):

                # Call lime if nonlinear
                if self.W0 is None and self.W is None:
                    # set seed for lime
                    np.random.seed(xi)
                    coefficients, intercept = lime_explanation(predict_proba_fn,
                                                               X_train, x)
                    coefficients, intercept = np.round_(coefficients, 4), np.round_(intercept, 4)
                    self.set_W(coefficients)
                    self.set_W0(intercept)

                    self.delta_max = d

                    r, _ = self.get_recourse(x, lamb)

                    self.set_W(None)
                    self.set_W0(None)
                else:
                    self.delta_max = d
                    r, _ = self.get_recourse(x, lamb)
                recourses.append(r)

            v = recourse_validity(predict_fn, recourses, target=self.y_target.numpy())
            if v >= v_old:
                v_old = v
            else:
                di = max(0, i - 1)
                return deltas[di]

        return d

    def choose_params(self, recourse_needed_X, predict_fn, X_train=None, predict_proba_fn=None):
        def get_validity(d, l, recourse_needed_X, predict_proba_fn):
            print("Testing delta %f, lambda %f" % (d, l))
            recourses = []
            for xi, x in tqdm(enumerate(recourse_needed_X)):

                self.delta_max = d

                # Call lime if nonlinear
                if self.W0 is None and self.W is None:
                    # set seed for lime
                    np.random.seed(xi)
                    coefficients, intercept = lime_explanation(predict_proba_fn,
                                                               X_train, x)
                    coefficients, intercept = np.round_(coefficients, 4), np.round_(intercept, 4)
                    self.set_W(coefficients)
                    self.set_W0(intercept)

                    r, _ = self.get_recourse(x, l)

                    self.set_W(None)
                    self.set_W0(None)

                else:
                    r, _ = self.get_recourse(x, l)

                recourses.append(r)
            v = recourse_validity(predict_fn, recourses, target=self.y_target.numpy())
            return v

        deltas = [0.01, 0.25, 0.5, 0.75]
        lambdas = [0.25, 0.5, 0.75, 1]

        m1_validity = np.zeros((4, 4))
        costs = np.zeros((4, 4))

        delta = None
        lamb = None
        for li, l in enumerate(lambdas):
            if li == 0:
                for di, d in enumerate(deltas):
                    d = deltas[di]
                    v = get_validity(d, l, recourse_needed_X, predict_proba_fn)
                    if v < m1_validity[max(0, di - 1)][li]:
                        di = max(0, di - 1)
                        delta = deltas[di]
                        break
                    m1_validity[di][li] = v

                if delta is None:
                    delta = d
            else:
                v = get_validity(delta, l, recourse_needed_X, predict_proba_fn)
                m1_validity[di][li] = v
                if v < m1_validity[di][max(0, li - 1)]:
                    li = max(0, li - 1)
                    lamb = lambdas[li]
                    break
        if lamb is None:
            lamb = l

        print(m1_validity)
        return delta, lamb


class CausalRecourse():
    def __init__(self, X_train, predict_proba_fn, torch_model,
                 robust=False,
                 W=None, W0=None, pW=None, pW0=None,
                 step_size=-1e2,
                 lamb=1,
                 delta_max=0.1,
                 feature_costs=None,
                 max_iter=100,
                 threshold=0.5,
                 target=1):
        self.X_train = X_train
        self.scm = GermanSCM(X_train)
        self.predict_proba_fn = predict_proba_fn
        self.torch_model = torch_model
        self.robust = robust
        self.W = W
        self.W0 = W0
        self.pW = pW
        self.pW0 = pW0
        self.step_size = step_size
        self.lamb = lamb
        self.delta_max = delta_max
        self.max_iter = 100
        self.y_target = 1
        self.feature_costs = feature_costs
        if self.feature_costs is not None:
            self.feature_costs = torch.from_numpy(feature_costs).float()

    def get_recourse(self, x, lime_seed=None):
        rec = x
        it = 0
        while it < self.max_iter:
            grad = self.get_grad(x, rec, lime_seed=None)
            nrec = self.scm.act(x, grad * self.step_size)
            rec = nrec
            f_rec = self.predict_proba_fn(rec.reshape(1, -1))[0][1]
            if f_rec >= 0.5:
                return rec
            it += 1
        return rec

    def get_grad(self, x, rec, lime_seed):
        x = torch.from_numpy(x).float()
        rec = Variable(torch.from_numpy(rec).float(), requires_grad=True)

        if self.feature_costs is None:
            cost = torch.dist(rec, x, 1)
        else:
            cost = torch.norm(self.feature_costs * (rec - x), 1)

        if self.robust:
            if self.W is None:
                np.random.seed(lime_seed)
                coefficients, intercept = lime_explanation(self.predict_proba_fn,
                                                           self.X_train, x.numpy())
                coefficients, intercept = np.round_(coefficients, 4), np.round_(intercept, 4)
                rr = RobustRecourse(W=coefficients, W0=intercept, delta_max=self.delta_max)
            else:
                rr = RobustRecourse(W=self.W, W0=self.W0, pW=self.pW, pW0=self.pW0, delta_max=self.delta_max)
            loss_fn = torch.nn.BCELoss()
            delta_W, delta_W0 = rr.calc_delta_opt(rec)
            delta_W, delta_W0 = torch.from_numpy(delta_W).float(), torch.from_numpy(delta_W0).float()
            f_x_new = torch.nn.Sigmoid()(torch.matmul(rr.W + delta_W, rec) + rr.W0 + delta_W0)[0]
            obj = loss_fn(f_x_new, torch.tensor(self.y_target).float()) + self.lamb * cost
        else:
            f_x_new = self.torch_model(rec)
            obj = self.lamb * (f_x_new - self.y_target) ** 2 + cost
        obj.backward()
        return rec.grad.detach().numpy()

    def choose_params(self, recourse_needed_X, predict_fn, choose_lambda=True):
        step_sizes = [-1e-3, -1e-2, -1e-1, -1, -10]
        lambdas = [1e-3, 1e-2, 1e-1, 1, 10]
        if choose_lambda == False:
            lambdas = [self.lamb]

        m1_validity = np.zeros((5, 5))
        costs = np.zeros((5, 5))

        for si, s in enumerate(step_sizes):
            for li, l in enumerate(lambdas):
                print("Testing step size %f, lambda %f" % (s, l))
                recourses = []
                for xi, x in tqdm(enumerate(recourse_needed_X)):
                    self.step_size = s
                    self.lamb = l
                    r, _ = self.get_recourse(x, lime_seed=xi)
                    recourses.append(r)
                v = recourse_validity(predict_fn, recourses, target=self.y_target)
                m1_validity[si][li] = v
                if self.feature_costs is None:
                    cost = l1_cost(recourse_needed_X, recourses)
                else:
                    cost = pfc_cost(recourse_needed_X, recourses, self.feature_costs)
                costs[si][li] = cost

        max_validity = np.amax(m1_validity)
        cand_indices = m1_validity >= max_validity
        min_cost = np.amin(costs[cand_indices])
        step_size_idxs, lamb_idxs = np.where(costs == min_cost)
        step_size, lamb = step_sizes[step_size_idxs[0]], lambdas[lamb_idxs[0]]

        return step_size, lamb
