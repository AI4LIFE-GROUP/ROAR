import argparse
from recourse_methods import *
from model import *
from recourse_utils import *
from data import * 
import pickle
from tqdm import tqdm 

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_trials', type=int, default=5, help='number of trials to run/experiment')
	parser.add_argument('--data', default="correction", help='which dataset to use')
	parser.add_argument('--base_model', default='lr',help='which model to use')
	parser.add_argument('--cost', default="l1", help='which cost fn to use')
	parser.add_argument('--lambdas', nargs='+', type=float,help='lambda param for robust_recourse')
	parser.add_argument('--recourse', default="robust", help='which recourse approach to use')
	args = parser.parse_args()

	result_fname = "_".join(["delta",args.recourse,args.data,args.base_model,args.cost])+".pkl"

	results = {}

	lambdas = args.lambdas
	if lambdas is None:
		lambdas = [0.1]*args.n_trials
	
	deltas = np.arange(0.1,1.1,0.1)
	for d in deltas:
		results_d = {}
		results_d["delta"] = d
		for i in range(args.n_trials):
			print("Delta %f, Trial %d" % (d,i))

			results_i = {}
			results_i["delta"] = d
			fold = i

			print("Loading %s dataset" % args.data)
			if args.data=="correction":
				data = CorrectionShift(fold)
				data1, data2 = data.get_data("datasets/german.csv", "datasets/corrected_german.csv")
			elif args.data=="temporal":
				data = TemporalShift(fold)
				data1, data2 = data.get_data("datasets/SBAcase.11.13.17.csv")
			elif args.data=="geospatial":
				data = GeospatialShift(fold)
				data1, data2 = data.get_data("datasets/student-por.csv", sep=";")
			
			X1_train, y1_train, X1_test, y1_test = data1
			X2_train, y2_train, X2_test, y2_test = data2

			print("Training %s models" % args.base_model)
			if args.base_model == "lr":
				m1 = LR()
				m2 = LR()
			if args.base_model == "nn":
				m1 = NN(X1_train.shape[1])
				m2 = NN(X1_train.shape[1])

			m1.train(X1_train.values, y1_train.values)
			m1_metrics = m1.metrics(X1_test.values, y1_test.values)
			results_i["m1_metrics"] = m1_metrics
			print("M1 Test acc:%f, Test AUC:%f" % m1_metrics)

			m2.train(X2_train.values, y2_train.values)
			m2_metrics = m2.metrics(X2_test.values, y2_test.values)
			results_i["m2_metrics"] = m2_metrics
			print("M2 Test acc:%f, Test AUC:%f" % m2_metrics)

			print("Finding where recourse is needing on X1_test")
			recourse_needed_idx_X1_test = recourse_needed(m1.predict, X1_test.values)
			recourse_needed_X1_test = X1_test.iloc[recourse_needed_idx_X1_test].values

			print("Using %s cost" % args.cost)
			if args.cost == "l1":
				feature_costs = None
			elif args.cost == "pfc":
				pfc = PFC(n_feat=X1_test.shape[1])
				feature_costs = pfc.get_costs()

			print("Getting %s recourse" % args.recourse)
			if args.recourse=="robust":

				coefficients=intercept=None
				
				if args.base_model=="lr":
					coefficients=m1.sklearn_model.coef_[0]
					intercept = m1.sklearn_model.intercept_
				
				robust_recourse = RobustRecourse(W=coefficients, 
					W0=intercept, feature_costs=feature_costs, delta_max=d)
				
				lamb = lambdas[i]
				print("Lambda", lamb)

				recourses=[]
				for xi, x in tqdm(enumerate(recourse_needed_X1_test)):
					if args.base_model!="lr":
						#set seed for lime
						np.random.seed(xi)
						coefficients, intercept = lime_explanation(m1.predict_proba, 
							X1_train.values, x)
						coefficients, intercept = np.round_(coefficients, 4), np.round_(intercept, 4)
						robust_recourse.set_W(coefficients)
						robust_recourse.set_W0(intercept)

					r = robust_recourse.get_recourse(x, lamb=lamb)
					recourses.append(r)
			
			elif args.recourse=="causal":
				coefficients, intercept, pW, pW0 = None, None, None, None
				if args.base_model!="nn":
					coefficients=m1.sklearn_model.coef_[0]
					intercept = m1.sklearn_model.intercept_
					
				if args.base_model=="svm":
					pW = m1.ps.coef_[0]
					pW0 = m1.ps.intercept_
				
				lamb = lambdas[i]
				print("Lambda", lamb)

				causal_recourse = CausalRecourse(X1_train, m1.predict_proba, m1.torch_model,
						feature_costs=feature_costs,robust=True,
						W=coefficients, W0=intercept,pW=pW, pW0=pW0, lamb=lamb, delta_max=d)

				print("Choosing step_size using X1_train")
				recourse_needed_idx_X1_train = recourse_needed(m1.predict, X1_train)
				recourse_needed_X1_train = X1_train.iloc[recourse_needed_idx_X1_train].values
				step_size, _ = causal_recourse.choose_params(recourse_needed_X1_train, 
					m1.predict,choose_lambda=False)
				results_i["step_size"] = step_size
				print("Chosen step_size:%f" % step_size)

				causal_recourse.step_size = step_size
				
				recourses=[]
				lime_seed = 0
				for x in tqdm(recourse_needed_X1_test):
					r = causal_recourse.get_recourse(x, lime_seed)
					lime_seed+=1
					recourses.append(r)

			results_i["recourses"] = recourses

			m1_validity = recourse_validity(m1.predict, recourses)
			results_i["m1_validity"] = m1_validity
			print("M1 validity: %f" % m1_validity)

			m2_validity = recourse_validity(m2.predict, recourses)
			results_i["m2_validity"] = m2_validity
			print("M2 validity: %f" % m2_validity)

			if args.cost == "l1":
				cost = l1_cost(recourse_needed_X1_test, recourses)
			elif args.cost == "pfc":
				cost = pfc_cost(recourse_needed_X1_test, recourses, feature_costs)

			results_i["cost"] = cost
			print("%s cost: %f" % (args.cost, cost))

			results_d[i] = results_i
		results[d] = results_d

	with open(result_fname, "wb") as f:
		pickle.dump(results, f) 








