import argparse
from recourse_methods import *
from model import *
from recourse_utils import *
from data import * 
import pickle
from tqdm import tqdm 
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_model', default='lr',help='which model to use')
	parser.add_argument('--cost', default="l1", help='which cost fn to use')
	parser.add_argument('--recourse', default="robust", help='which recourse approach to use')
	parser.add_argument('--lamb', default=0.1, type=float, help='lambda param for robust_recourse')
	parser.add_argument('--delta', default=None, type=float)
	parser.add_argument('--n_trials', type=int, default=5, help='number of trials to run/experiment')
	args = parser.parse_args()

	result_fname = "_".join(["simdata",args.base_model,args.cost,args.recourse])+".pkl"
	results = {}
	
	for i in range(args.n_trials):
		print("Trial %d" % i)

		results_i = {}
		seed = i
		
		simdata = SimulatedData(seed)
		X_train, X_test, y_train, y_test = simdata.get_data(num_samples=10000)


		print("Training %s models" % args.base_model)
		if args.base_model == "lr":
			m = LR()
		if args.base_model == "svm":
			m = SVM()
		if args.base_model == "nn":
			m = NN(X_train.shape[1])

		m.train(X_train, y_train)
		m_metrics = m.metrics(X_test, y_test)
		results_i["m_metrics"] = m_metrics
		print("Test acc:%f, Test AUC:%f" % m_metrics)

		print("Finding where recourse is needing on X_test")
		recourse_needed_idx_X_test = recourse_needed(m.predict, X_test.values)
		recourse_needed_X_test = X_test.iloc[recourse_needed_idx_X_test].values

		print("Using %s cost" % args.cost)
		if args.cost == "l1":
			feature_costs = None
		elif args.cost == "pfc":
			pfc = PFC(n_feat=X_test.shape[1])
			feature_costs = pfc.get_costs()

		print("Getting %s recourse" % args.recourse)
		if args.recourse=="robust":

			coefficients=intercept=None
			
			if args.base_model!="nn":
				coefficients=m.sklearn_model.coef_[0]
				intercept = m.sklearn_model.intercept_
			
			robust_recourse = RobustRecourse(W=coefficients, 
				W0=intercept, feature_costs=feature_costs, delta_max=args.delta)
			
			if args.base_model=="svm":
				robust_recourse.set_pW(m.ps.coef_[0])
				robust_recourse.set_pW0(m.ps.intercept_)

			if args.delta is None:
				print("Choosing hyperparameters delta and lambda using X_train")
				recourse_needed_idx_X_train = recourse_needed(m.predict, X_train)
				recourse_needed_X_train = X_train[recourse_needed_idx_X_train]
				delta, lamb = robust_recourse.choose_params(recourse_needed_X_train, 
					m.predict, X_train, m.predict_proba)
				# delta = robust_recourse.choose_delta(recourse_needed_X_train, 
				# 	m.predict, X_train, m.predict_proba)
				results_i["delta"] = delta
				results_i["lambda"] = lamb
				robust_recourse.delta_max = delta
				print("Chosen delta:%f" % delta)
				print("Chosen lamb:%f" % lamb)


			recourses=[]
			deltas=[]
			for xi, x in tqdm(enumerate(recourse_needed_X_test)):
				if args.base_model=="nn":
					#set seed for lime
					np.random.seed(xi)
					coefficients, intercept = lime_explanation(m.predict_proba, 
						X_train, x)
					coefficients, intercept = np.round_(coefficients, 4), np.round_(intercept, 4)
					robust_recourse.set_W(coefficients)
					robust_recourse.set_W0(intercept)

				r, delta_r = robust_recourse.get_recourse(x, lamb=args.lamb)
				recourses.append(r)
				deltas.append(delta_r)

		elif args.recourse=="actionable":
			X_train_df = pd.DataFrame({"X1":X_train[:,0],"X2":X_train[:,1]})
			recourses=[]
			for xi, x in tqdm(enumerate(recourse_needed_X_test)):

				if args.base_model=="nn":
					#set seed for lime
					np.random.seed(xi)
					coefficients, intercept = lime_explanation(m.predict_proba, 
						X_train, x)
					coefficients, intercept = np.round_(coefficients, 4), np.round_(intercept, 4)
				else:
					coefficients, intercept = m.sklearn_model.coef_[0], m.sklearn_model.intercept_
				'''
				r = actionable_recourse(x,X_train_df, coefficients=coefficients, 
						intercept=intercept[0], cost_type=args.cost, 
						feature_costs=feature_costs)
				'''
				r=0
				recourses.append(r)

		elif args.recourse=="counterfactual":
			recourses=[]
			for x in tqdm(recourse_needed_X_test):
				r = counterfactual_recourse(m.torch_model, x, feature_costs)
				recourses.append(r)

		results_i["recourses"] = recourses
		if args.recourse =="robust":
			results_i["delta_vec"] = deltas

		if args.cost == "l1":
			cost = l1_cost(recourse_needed_X_test, recourses)
		elif args.cost == "pfc":
			cost = pfc_cost(recourse_needed_X_test, recourses, feature_costs)

		results_i["cost"] = cost
		results_i["model"] = m
		print("%s cost: %f" % (args.cost, cost))

		v = recourse_validity(m.predict, recourses)
		results_i["validity"] = v
		print("Recourse recourse_validity: %f" % v)

		results[i] = results_i

	with open(result_fname, "wb") as f:
		pickle.dump(results, f) 


