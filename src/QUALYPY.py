###############
## Libraries ##
###############

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.nonparametric.smoothers_lowess import lowess

####################################################################
# FUNCTION RUN_EXAMPLE_QUALYPY
####################################################################
def run_example_QUALYPY():
    # prepare inputs of the data: a single pandas.DataFrame where
    # - there is one column corresponding to the predictand (e.g. annual precipitation amounts for a future period)
    # - the is one optional column corresponding to the continuous predictor (time, warming level)
    # - there are at least one categorical predictor corresponding to the type of simulations (type of GCM, etc.) that will be used to apply
    # the fixed-effect ANOVA model

    # Consider known marginal effects for 5 RCMs and 2 GCMs. The sum of the effects by type of model is equal to zero
    RCMeffect = {
        "RCM1":1,
        "RCM2":-1,
        "RCM3":3,
        "RCM4":4,
        "RCM5":-7
        }

    GCMeffect = {
        "GCM1":5,
        "GCM2":-5
    }

    # get vector of RCMs/GCMs
    vecRCM  =list(RCMeffect.keys())
    vecGCM  =list(GCMeffect.keys())

    # number of years i.e. the "predictor"
    nY = 100
    x = np.array(range(nY))

    # Loop over the different combinations of GCMs and RCMs and generate a fake ensemble
    # of projections. Each projection is a time series of the predictand Y
    # for a given RCM and GCM combination.
    list_df = []
    i=-1
    for GCM in vecGCM:
        for RCM in vecRCM:
            i = i+1
            
            # generate a synthetic projection Y for a combination of a RCM and a GCM
            # - it has a mean of 5+0.05*x (=10 at the end of the series)
            # - an interannual standard deviation of 2
            # - combined effects between -26 and 9 at the end of the series
            y_i = np.random.default_rng().normal(5+x*0.05,2,nY)+(RCMeffect[RCM]+GCMeffect[GCM])*x*0.02
            
            # build data.frame. 
            data = {
            'RCM': [RCM]*nY,
            'GCM': [GCM]*nY,
            'years': x+2000,
            'Y': y_i
            }
            
            # Create a DataFrame from the dictionary
            df = pd.DataFrame(data)
            list_df.insert(i,df)

    dfTest = pd.concat(list_df)


    ## LIST OF ARGUMENTS: head of the main function, add possible arguments here

    # name of the predictand
    Y = "Y"

    # name of the continuous predictor
    X = "years"

    # name of the categorical factors
    name_effect = ["RCM", "GCM"]

    # value of the continuous predictor for the future period
    Xpred = [2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]

    # type of change: "absolute" or "relative"
    type_change = "absolute"

    list_anova_input = get_ccr_mme_lowess(dfTest,name_effect,type_change,X,Y,Xpred)
    
    # FIGURE 1: Climate responses
    plt.figure(figsize=(10, 6))
    for anova_input in list_anova_input:
        plt.plot(anova_input["X"],anova_input["CS"], linestyle=':')
    for anova_input in list_anova_input:
        plt.plot(anova_input["X"],anova_input["CRFIT"], linestyle='-')
    plt.title("Figure 1")
    
    list_anova_output = apply_anova_mme(dfTest,name_effect,type_change,X,Y,Xpred)
    
    # FIGURE 2: DECOMPOSITION OF THE TOAL VARIANCE 
    plt.figure(figsize=(10, 6))
    list_anova_output["TOTAL_VARIANCE_PERCENTAGE"].plot.area()
    plt.title("Figure 2")

    # FIGURE 3: MAIN EFFECTS
    plt.figure(figsize=(10, 6))
    for te in list_anova_output["MAIN_EFFECTS"]:
        list_anova_output["MAIN_EFFECTS"][te].plot()
    plt.title("Figure 3")
    
    plt.show()

####################################################################
# FUNCTION CLEAN_Y
####################################################################
def clean_y(y):
    """
        return array where nan values have been replaced by zero values, and weights indicating
        where these values hqve been replaced    
    
		Parameters
		---------
		y: np.array[ shape = (n_samples) ]
			Raw time series
		"""

    # Identify NaN values
    nan_mask = np.isnan(y)

    # Create weights: 0 for NaN locations, 1 for valid data
    weights = np.ones_like(y, dtype=float)
    weights[nan_mask] = 0.

    # Replace NaN values in y (e.g., with 0)
    y_cleaned = np.copy(y)
    y_cleaned[nan_mask] = 0.
    
    return(y_cleaned, weights)


####################################################################
# FUNCTION GET_CCR_MME
####################################################################

def get_ccr_mme_lowess(df,name_effect,type_change,X,Y,Xpred):
    """
    Extract climate responses (CR), climate changes responses (CCR), and deviations (DEV_CCR) from these climate change responses
    due to internal variability. The climate response is obtained using smoothing splines applied to each raw climate simulation.
    The smoothing parameter is chosen to minimize the number of non-monotonic changes in the climate response. It is iteratively
    increasing it until the number of non-monotonic changes in the climate response is less than or equal to 2:
    - CR: The climate response is defined as the fitted values of a smoothing spline applied to the raw climate simulation.
    - CRFIT: The climate response fitted by a smoothing spline for this scenario.
    - CRPRED: The climate response predicted by a smoothing spline for the future period.
    - CCR: The climate change response is computed as the difference between the climate response for a given scenario and the reference climate response.
    - DEV_CCR: The deviation from the climate response is computed as the difference between the raw climate simulation and the fitted climate response.

    Parameters
    ---------
    - df: a single pandas.DataFrame where:
    * there is one column corresponding to the predictand (e.g. annual precipitation amounts for a future period)
    * the is one optional column corresponding to the continuous predictor (time, warming level)
    * there are at least one categorical predictor corresponding to the type of simulations (type of GCM, etc.) 
    that will be used to apply the fixed-effect ANOVA model
    - name_effect: list of strings
        names of the categorical factors that will be used to apply the fixed-effect ANOVA model,
    - type_change: string
        type of change to compute: "absolute" or "relative"
    - X: string
        name of the continuous predictor in the DataFrame df, e.g. "years"
    - Y: string
        name of the predictand in the DataFrame df, e.g. "Y"
    - Xpred: list of floats
        values of the continuous predictor for the future period, 
        e.g. [2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
        
    
    Returns
    - list_anova_input: list of dictionaries ordered by the categorical factors
        each dictionary contains the following keys:
        - "factors": tuple of the categorical factors for this scenario
        - "X": array of the continuous predictor for this scenario
        - "CS": array of the raw climate simulation for this scenario
        - "CRFIT": array of the climate response fitted by a smoothing spline for this scenario. The climate
        response is defined as the fitted values of a smoothing spline applied to the raw climate simulation.
        - "CRPRED": array of the climate response predicted by a smoothing spline for the future period
        - "CCR": array of the climate change response for this scenario. The climate change response is 
        defined as the difference between the climate response for a given scenario and the reference climate response
    (the climate response for the first value of Xpred).
        - "DEV_CCR": array of the deviation from the climate response due to interannual variability for this scenario.
        The deviation from the climate response is defined as the difference between the raw climate simulation and the 
        fitted climate response.
        """

    # Scherrer et al. 2024 advise for lowess smoothing for the extraction
    # of long-term climate responses: 10.1016/j.cliser.2023.100428
    # they suggest a fixed parameter for the degree of smoothing which
    # corresponses to a local window of 42 years for the local smoothing
    nyloess = 42

    # process ensemble: array nS x nY of projections
    df_effect_type_raw = df.drop_duplicates(subset=name_effect)
    df_effect_type = df_effect_type_raw[name_effect]

    # step 1 and 2: apply smoothing splines to the scenarios
    # and compute climate change responses
    list_anova_input = list()

    # for each scenario, list_anova_input will contain      
    # - factors: combination of categorical factors
    # - CS: raw climate simulation
    # - CR: climate response
    # - CCR: climate change response
    # - DEV_CCR: deviation from the climate response due to interannual variability
    for factors,subdf in df.groupby(name_effect):
        # extract continuous predictand/predictor
        x_i = subdf[X]
        y_i, w_i = clean_y(subdf[Y])
        
        # lowess smoothing (Scherrer et al., 2024)
        lowess_smoothed = lowess(y_i, x_i, frac=nyloess/len(y_i), it=0)
        cr_fitted = lowess_smoothed[:, 1]
        cr_predicted = lowess(y_i, x_i, frac=nyloess/len(y_i), xvals=Xpred, it=0)
        cr_ref = cr_predicted[0]
        
        # climate change response ccr_i and deviation from the climate response dev_ccr_i
        if(type_change=="absolute"):
            ccr_i = cr_predicted-cr_ref
            dev_ccr_i = y_i-cr_fitted
        else:
            ccr_i = cr_predicted/cr_ref-1
            dev_ccr_i = (y_i-cr_fitted)/cr_ref
        
        anova_input = {"factors": factors, "X": x_i, "CS": y_i, "CRFIT": cr_fitted, "CRPRED": cr_predicted, "CCR": ccr_i, "DEV_CCR": dev_ccr_i}
        
        list_anova_input.append(anova_input)
        
    return(list_anova_input)
    
    
####################################################################
# FUNCTION APPLY_ANOVA_MME
####################################################################

def apply_anova_mme(df,name_effect,type_change,X,Y,Xpred):
    """
    Extract climate responses (CR), climate changes responses (CCR), and deviations (DEV_CCR) from these climate change responses
    due to internal variability. The climate response is obtained using smoothing splines applied to each raw climate simulation.
    The smoothing parameter is chosen to minimize the number of non-monotonic changes in the climate response. It is iteratively
    increasing it until the number of non-monotonic changes in the climate response is less than or equal to 2:
    - CR: The climate response is defined as the fitted values of a smoothing spline applied to the raw climate simulation.
    - CRFIT: The climate response fitted by a smoothing spline for this scenario.
    - CRPRED: The climate response predicted by a smoothing spline for the future period.
    - CCR: The climate change response is computed as the difference between the climate response for a given scenario and the reference climate response.
    - DEV_CCR: The deviation from the climate response is computed as the difference between the raw climate simulation and the fitted climate response.

    Parameters
    ---------
    - df: a single pandas.DataFrame where:
    * there is one column corresponding to the predictand (e.g. annual precipitation amounts for a future period)
    * the is one optional column corresponding to the continuous predictor (time, warming level)
    * there are at least one categorical predictor corresponding to the type of simulations (type of GCM, etc.) 
    that will be used to apply the fixed-effect ANOVA model
    - name_effect: list of strings
        names of the categorical factors that will be used to apply the fixed-effect ANOVA model,
    - type_change: string
        type of change to compute: "absolute" or "relative"
    - X: string
        name of the continuous predictor in the DataFrame df, e.g. "years"
    - Y: string
        name of the predictand in the DataFrame df, e.g. "Y"
    - Xpred: list of floats
        values of the continuous predictor for the future period, 
        e.g. [2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
        
    
    Returns
    - anova_input: output of the function get_ccr_mme
    - VARIANCE_COMPONENT: pandas.DataFrame
        variance components for each time in Xpred, with columns:
        - "INTERNAL": internal variability
        - "RESIDUAL": residual variability
        - name_effect: variance contributions of each categorical factor
    - TOTAL_VARIANCE_PERCENTAGE: pandas.DataFrame
        contribution (in %) to the total variance for each time in Xpred, with columns:
        - "INTERNAL": internal variability
        - "RESIDUAL": residual variability
        - name_effect: variance contributions of each categorical factor
    - MAIN_EFFECTS: dictionary of pandas.DataFrame
        main effects for each categorical factor, with columns:
        - name of the factor: main effects for each value of the factor
    - CONTRIBUTION_MAIN_EFFECTS: dictionary of pandas.DataFrame
        contribution of each main effect to the variance for each time in Xpred, with columns:
        - name of the factor: contribution of each value of the factor to the variance for each time in Xpred
        """
  
    list_anova_input = get_ccr_mme_lowess(df,name_effect,type_change,X,Y,Xpred)

    list_anova_output = list()

    #_______________________________________
    #           INTERNAL VARIABILITY
    #_______________________________________
    # create list of tuples
    vec_dev_ccr = []
    for anova_input in list_anova_input:
        vec_dev_ccr.append(anova_input["DEV_CCR"])
        
    INTERNAL_VAR = np.mean(np.array(vec_dev_ccr)**2)

    #_______________________________________
    #                     ANOVA
    #_______________________________________

    residual_var = []
    for t in range(len(Xpred)):
        list_anova_output_t = list()
        
        # create list
        df_ANOVA_unset = []
        for anova_input in list_anova_input:
            df_ANOVA_unset.append(anova_input["factors"] + (anova_input["CCR"][t],))

        # convert to DataFrame
        column_names = name_effect.copy()
        column_names.append("Y")
        df_ANOVA = pd.DataFrame(df_ANOVA_unset, columns=column_names)

        # build generic formula for any number of effects
        formula = 'Y ~ '
        for eff in name_effect:
            formula = formula + '+C(' + eff + ', Sum)'

        # apply ANOVA model
        lm = ols(formula, data=df_ANOVA).fit()
        
        # residual variability for this Xpred
        residual_var.append(np.var(lm.resid))
        
        # start a dictionary here
        list_anova_output_t = {}
        
        list_factors = {}
        
        for te in name_effect:
            # array of unique factor values
            vec_factor = df_ANOVA[te].unique()
            list_factors[te] = vec_factor.copy()
            
            # reconstruct individual effects: retrieve estimates in lm outputs
            # by constraint, the last effect corresponds to the oppositive sum of the other ones
            # and is not produced in the outputs. We calculate it here.
            eff_estimates = {}
            sumEff = 0
            for factor in vec_factor:
                lm_param_name = 'C('+te+', Sum)[S.' + factor +']'
                if lm_param_name in lm.params:
                    eff = lm.params[lm_param_name]
                    sumEff = sumEff + eff
                    eff_estimates[factor] = eff
                else:
                    missing_factor = factor
                
            # prepare outputs obtained for each categorical variable
            # main_effects_factor contains the main effects (len(vec_factor))
            eff_estimates[missing_factor] = -sumEff
            main_effects_factor = np.array(list(eff_estimates.values()))
            
            # eff_var is the corresponding variance
            eff_var = np.var(main_effects_factor)
            
            # contrib_eff_var is the contribution of each factor to the corresponding variance  (len(vec_factor))
            # e.g. the contribution of each individual RCM to the RCM variance
            contrib_eff_var = main_effects_factor**2/eff_var/len(vec_factor)
            
            # add outputs for this effect
            list_anova_output_t[te] = {"main_effects_factor": main_effects_factor, "eff_var": eff_var,"contrib_eff_var": contrib_eff_var}
        
        # add all outputs obtained for this time t
        list_anova_output.append(list_anova_output_t)

    #_______________________________________
    #           FORMAT OUTPUTS
    #_______________________________________

    VARIANCE_COMPONENT_LIST = []

    ###########################################
    # TOTAL VARIANCE AND VARIANCE CONTRIBUTIONS
    ###########################################
    for t in range(len(Xpred)):
        # internal and residual variabilities
        vec_var = [INTERNAL_VAR, residual_var[t]]
        
        # add effect variances
        for te in name_effect:
            vec_var.append(list_anova_output[t][te]["eff_var"])
        
        VARIANCE_COMPONENT_LIST.append(vec_var)
        
    VARIANCE_COMPONENT_VSTACK = np.vstack(VARIANCE_COMPONENT_LIST)

    column_names = ["INTERNAL","RESIDUAL"]+name_effect.copy()

    VARIANCE_COMPONENT = pd.DataFrame(VARIANCE_COMPONENT_VSTACK, columns=column_names, index=Xpred)

    # Total variance
    TOTAL_VARIANCE = VARIANCE_COMPONENT.sum(axis=1)
    TOTAL_VARIANCE_PERCENTAGE = VARIANCE_COMPONENT.div(TOTAL_VARIANCE, axis=0)


    ###########################################
    # MAIN EFFECTS AND INDIVIDUAL CONTRIBUTIONS
    ###########################################
    MAIN_EFFECTS_VSTACK = {}
    MAIN_EFFECTS = {}
    CONTRIBUTION_MAIN_EFFECTS_VSTACK = {}
    CONTRIBUTION_MAIN_EFFECTS = {}
    for te in name_effect:
        MAIN_EFFECTS_VSTACK[te] = []
        CONTRIBUTION_MAIN_EFFECTS_VSTACK[te] = []
        
        for t in range(len(Xpred)):
            MAIN_EFFECTS_VSTACK[te].append(list_anova_output[t][te]["main_effects_factor"])
            CONTRIBUTION_MAIN_EFFECTS_VSTACK[te].append(list_anova_output[t][te]["contrib_eff_var"])
            
            
        MAIN_EFFECTS[te] = pd.DataFrame(MAIN_EFFECTS_VSTACK[te], columns=list_factors[te].copy(), index=Xpred)
        CONTRIBUTION_MAIN_EFFECTS[te] = pd.DataFrame(CONTRIBUTION_MAIN_EFFECTS_VSTACK[te], 
                                                     columns=list_factors[te].copy(), index=Xpred)
    
    ###########################################
    # FINAL OUTPUTS
    ###########################################
    OUT = {"anova_input": list_anova_input,"CONTRIBUTION_MAIN_EFFECTS": CONTRIBUTION_MAIN_EFFECTS, "MAIN_EFFECTS": MAIN_EFFECTS,
        "VARIANCE_COMPONENT": VARIANCE_COMPONENT, "TOTAL_VARIANCE_PERCENTAGE": TOTAL_VARIANCE_PERCENTAGE}
    
    return(OUT)