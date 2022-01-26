import argparse
from recourse_methods import *
from model import *
from recourse_utils import *
from data import * 
import pickle
from tqdm import tqdm 
import ast

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_trials', type=int, default=5, help='number of trials to run/experiment')
	parser.add_argument('--data', default="correction", help='which dataset to use')
	parser.add_argument('--base_model', default='lr',help='which model to use')
	parser.add_argument('--cost', default="l1", help='which cost fn to use')
	parser.add_argument('--recourse', default="robust", help='which recourse approach to use')
	parser.add_argument('--lamb', default=None, type=float,help='lambda param for robust_recourse')
	parser.add_argument('--causal_robust', default=False, type=ast.literal_eval,help='ROAR-MINT vs MINT flag')
	args = parser.parse_args()

	result_fname = "_".join([args.data,args.base_model,args.cost,args.recourse,str(args.n_trials)])+".pkl"
	if args.recourse=="causal":
		result_fname = "_".join([args.data,args.base_model,args.cost,
			args.recourse,str(args.n_trials),str(args.causal_robust)])+".pkl"
	results = {}
	
	for i in range(args.n_trials):
		print("Trial %d" % i)

		results_i = {}
		fold = i

		print("Loading %s dataset" % args.data)
		if args.data=="correction":
			data = CorrectionShift(fold)
			data1, data2 = data.get_data("datasets/german.csv", "datasets/corrected_german.csv")
			#carla_data = CorrectionShiftCarla(fold,"datasets/german.csv", "datasets/corrected_german.csv")
		elif args.data=="temporal":
			data = TemporalShift(fold)
			data1, data2 = data.get_data("datasets/SBAcase.11.13.17.csv")
			#carla_data = TemporalShiftCarla(fold, "datasets/SBAcase.11.13.17.csv")
		elif args.data=="geospatial":
			data = GeospatialShift(fold)
			data1, data2 = data.get_data("datasets/student-por.csv", sep=";")
			#carla_data = GeospatialShiftCarla(fold,"datasets/student-por.csv",";")
		
		X1_train, y1_train, X1_test, y1_test = data1
		X2_train, y2_train, X2_test, y2_test = data2

		print("Training %s models" % args.base_model)
		if args.base_model == "lr":
			m1 = LR()
			m2 = LR()
		if args.base_model == "nn":
			m1 = NN(X1_train.shape[1])
			m2 = NN(X1_train.shape[1])
		if args.base_model == "svm":
			m1 = SVM()
			m2 = SVM()

		m1.train(X1_train.values, y1_train.values)
		m1_metrics = m1.metrics(X1_test.values, y1_test.values)
		results_i["m1_metrics"] = m1_metrics
		print("M1 Test acc:%f, Test AUC:%f" % m1_metrics)

		#carla m1
		#carla_model = CarlaModel(carla_data, m1)

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
			
			if args.base_model!="nn":
				coefficients=m1.sklearn_model.coef_[0]
				intercept = m1.sklearn_model.intercept_
			
			robust_recourse = RobustRecourse(W=coefficients, 
				W0=intercept, feature_costs=feature_costs)

			if args.base_model=="svm":
				robust_recourse.set_pW(m1.ps.coef_[0])
				robust_recourse.set_pW0(m1.ps.intercept_)
			
			if args.lamb is None:
				print("Choosing hyperparameter lambda using X1_train")
				# robust_recourse = RobustRecourse(W=coefficients, 
				# 	W0=intercept, feature_costs=feature_costs)
				recourse_needed_idx_X1_train = recourse_needed(m1.predict, X1_train)
				recourse_needed_X1_train = X1_train.iloc[recourse_needed_idx_X1_train].values
				lamb = robust_recourse.choose_lambda(recourse_needed_X1_train, 
					m1.predict, X1_train.values, m1.predict_proba)
				results_i["lambda"] = lamb
				print("Chosen lambda:%f" % lamb)
			else:
				lamb = args.lamb

			recourses=[]
			deltas=[]
			for xi, x in tqdm(enumerate(recourse_needed_X1_test)):
				if args.base_model=="nn":
					#set seed for lime
					np.random.seed(xi)
					coefficients, intercept = lime_explanation(m1.predict_proba, 
						X1_train.values, x)
					coefficients, intercept = np.round_(coefficients, 4), np.round_(intercept, 4)
					robust_recourse.set_W(coefficients)
					robust_recourse.set_W0(intercept)
				r, delta_r = robust_recourse.get_recourse(x, lamb=lamb)
				recourses.append(r)
				deltas.append(delta_r)

		elif args.recourse=="actionable":
			recourses=[]
			for xi, x in tqdm(enumerate(recourse_needed_X1_test)):
				if args.base_model=="nn":
					#set seed for lime
					np.random.seed(xi)
					coefficients, intercept = lime_explanation(m1.predict_proba, 
						X1_train.values, x)
					coefficients, intercept = np.round_(coefficients, 4), np.round_(intercept, 4)
				else:
					coefficients, intercept = m1.sklearn_model.coef_[0], m1.sklearn_model.intercept_
				r = actionable_recourse(x,X1_train, coefficients=coefficients, 
						intercept=intercept[0], cost_type=args.cost, 
						feature_costs=feature_costs)
				
				recourses.append(r)

		elif args.recourse=="counterfactual":
			recourses=[]
			for x in tqdm(recourse_needed_X1_test):
				r = counterfactual_recourse(m1.torch_model, x, feature_costs)
				recourses.append(r)

		elif args.recourse=="causal":
			coefficients, intercept, pW, pW0 = None, None, None, None
			if args.base_model!="nn":
				coefficients=m1.sklearn_model.coef_[0]
				intercept = m1.sklearn_model.intercept_
				
			if args.base_model=="svm":
				pW = m1.ps.coef_[0]
				pW0 = m1.ps.intercept_
				
			causal_recourse = CausalRecourse(X1_train, m1.predict_proba, m1.torch_model,
					feature_costs=feature_costs,robust=args.causal_robust,
					W=coefficients, W0=intercept,pW=pW, pW0=pW0)

			print("Choosing hyperparameters using X1_train")
			recourse_needed_idx_X1_train = recourse_needed(m1.predict, X1_train)
			recourse_needed_X1_train = X1_train.iloc[recourse_needed_idx_X1_train].values
			step_size, lamb = causal_recourse.choose_params(recourse_needed_X1_train, m1.predict)
			results_i["step_size"] = step_size
			results_i["lambda"] = lamb
			print("Chosen step_size:%f, lambda:%f" % (step_size, lamb))

			causal_recourse.step_size = step_size
			causal_recourse.lamb = lamb
			
			recourses=[]
			lime_seed = 0
			for x in tqdm(recourse_needed_X1_test):
				r = causal_recourse.get_recourse(x, lime_seed)
				lime_seed+=1
				recourses.append(r)
		
		# elif args.recourse == "cchvae":
		# 	n_feat = len(carla_model.feature_input_order)
		# 	cchvae = CCHVAE(carla_model, hyperparams={"data_name": args.data,
		# 							 "pnorm":1,
		# 							  "clamp":False,
		# 							  "step":0,
		# 							  "binary_cat_features":True,
		# 							  "vae_params":{"layers":[n_feat]+[200]*5+[int(0.5*n_feat)],
		# 											"epochs":100,
		# 											"lr":1e-3}})
		# 	factuals = X1_test.iloc[recourse_needed_idx_X1_test]
		# 	factuals[carla_data.target] = np.zeros(len(factuals))
		# 	recourses = cchvae.get_counterfactuals(factuals).drop(columns=[carla_data.target]).values
			

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

		results[i] = results_i

		results_i["recourses"] = recourses
		if args.recourse =="robust":
			results_i["delta_vec"] = deltas
			
	with open(result_fname, "wb") as f:
		pickle.dump(results, f) 

	agg_m1_validity = []
	agg_m2_validity = []
	agg_cost = []
	for i in range(args.n_trials):
		agg_m1_validity.append(results[i]["m1_validity"])
		agg_m2_validity.append(results[i]["m2_validity"])
		agg_cost.append(results[i]["cost"])

	print("Average M1 validity: %f +- %f" % (np.mean(agg_m1_validity), np.std(agg_m1_validity)))
	print("Average M2 validity: %f +- %f" % (np.mean(agg_m2_validity), np.std(agg_m2_validity)))
	print("Average cost: %f +- %f" % (np.mean(agg_cost), np.std(agg_cost)))








