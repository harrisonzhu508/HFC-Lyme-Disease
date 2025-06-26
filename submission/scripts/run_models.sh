for seed in 0 1 2 3 4;
    do 
        # python scripts/run_gpr.py --seed=$seed 
        # python scripts/run_lasso_conformal.py --seed=$seed 
        # python scripts/run_gpr_svgp.py --seed=$seed 
        python scripts/run_xgboost.py --seed=$seed 
    done