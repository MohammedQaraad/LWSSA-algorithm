# LWSSA-algorithm
Fine-Tuned Cardiovascular Risk Assessment: Locally-Weighted Salp Swarm Algorithm in Global Optimization

The Salp swarm algorithm (SSA) is a bio-inspired metaheuristic optimization technique that mimics the collective behavior of Salp chains hunting for food in the ocean. While demon-strating competitive performance on benchmark problems, SSA faces challenges with slow con-vergence and getting trapped in local optima like many population-based algorithms. To address these limitations, this study proposes the locally weighted Salp swarm algorithm (LWSSA), which combines two mechanisms into the standard SSA framework. First, a locally weighted approach is introduced and integrated into SSA to guide the search toward locally promising regions. This heuristic iteratively probes high-quality solutions in the neighborhood and refines the current po-sition. Second, a mutation operator generates new positions for Salp followers to increase ran-domness throughout the search. In order to assess its effectiveness, the proposed approach was evaluated against the state-of-the-art metaheuristics using standard test functions from IEEE CEC 2021 and IEEE CEC 2017 competitions. The methodology is also applied to risk assessment of car-diovascular disease (CVD). Seven optimization strategies of the extreme Gradient Boosting (XGBoost) classifier are evaluated and compared to the proposed LWSSA-XGBoost model. The proposed LWSSA-XGBoost achieves superior prediction performance with 94% F1-score, 94% re-call, 93% accuracy, and 93% area under the ROC curve in comparison with state-of-art competitors. This provides clinicians with a more robust foundation for evaluating patient health risks. Overall, the experimental results demonstrate that LWSSA enhances SSA's optimization ability and XGBoost predictive power in automated CVD risk assessment.
