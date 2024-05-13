
# Import packages
import os

# Import packages
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import patsy as pt

from scipy.interpolate import interp1d
from scipy.stats import median_test, shapiro
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tabulate import tabulate

from .cross_validation_poly_features import one_standard_error_rule, bootstrap_poly_ols_analysis, poly_order_cv

k = 5
poly_order_max = 5

demographics_colors = ["#5e3c99", "#f1a340", "#1b9e77"]
FIGURES_DIR = os.path.join("..", "figures")

# Linear Regression function
def ols(df, variable_of_interest, response_of_interest):
    X = df[variable_of_interest].values
    y = df[response_of_interest].values
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const)
    results = model.fit()
    return results


# Fit an OLS model to a single Bootstrap sample
def bootstrap_ols_sample_analysis(X, y, x_array):
    x_bootstrap_min = min(X)
    x_bootstrap_max = max(X)

    X_reshape = np.array(X).reshape(-1, 1)

    # Fit the polynomial regression
    regr = linear_model.LinearRegression()
    regr.fit(X_reshape, y)

    # You do no want to extrapolate the bootstrapped regression beyond the range of the bootstrap data
    # For values below the lowest income in the bootstrap sample and above the highest income in the
    # bootstrap sample, place NaN values.
    entries_below_bootstrap_min = sum(x_array < x_bootstrap_min)
    y_pred_below_bootstrap_min = np.full(entries_below_bootstrap_min, np.nan)

    entries_above_bootstrap_max = sum(x_array > x_bootstrap_max)
    y_pred_above_bootstrap_max = np.full(entries_above_bootstrap_max, np.nan)

    # For entries within the bootstrap sample range, used your fitted regression to predict solar installation
    values_within_bootstrap_data = x_array[
        entries_below_bootstrap_min : (
            len(x_array) - entries_above_bootstrap_max
        )
    ]
    values_within_bootstrap_data = values_within_bootstrap_data.reshape(-1, 1)
    y_pred_within_bootstrap_data_range = regr.predict(
        values_within_bootstrap_data
    )

    # Concatenate the predicted solar installations for all three regions (below the lowest income in the bootstrap
    # sample, within the income data in the bootstrap sample, and above the highest income in the bootstrap sample)
    y_pred = np.concatenate(
        (
            y_pred_below_bootstrap_min,
            y_pred_within_bootstrap_data_range,
            y_pred_above_bootstrap_max,
        )
    )

    return y_pred


# Bootstrap function to get confidence interval for OLS models
def bootstrap_ols_analysis(df, variable_of_interest, response_of_interest):
    X = df[variable_of_interest]
    X = X.reset_index(drop=True)
    y = df[response_of_interest]
    y = y.reset_index(drop=True)

    x_array = np.linspace(min(X),max(X),100)

    y_pred_bootstrap = pd.DataFrame()
    for i in range(0, 100):
        df_bootstrap = df.sample(n=len(df), replace=True)
        X_bootstrap = df_bootstrap[variable_of_interest]
        X_bootstrap = X_bootstrap.reset_index(drop=True)
        y_bootstrap = df_bootstrap[response_of_interest]
        y_bootstrap = y_bootstrap.reset_index(drop=True)

        y_pred = bootstrap_ols_sample_analysis(
            X_bootstrap, y_bootstrap, x_array
        )
        y_pred_bootstrap[i] = y_pred

    y_pred_bootstrap_summary = pd.DataFrame()
    y_pred_bootstrap_summary["median"] = np.nanpercentile(
        y_pred_bootstrap, 50, axis=1
    )
    y_pred_bootstrap_summary["95_CI_high"] = np.nanpercentile(
        y_pred_bootstrap, 97.5, axis=1
    )
    y_pred_bootstrap_summary["95_CI_low"] = np.nanpercentile(
        y_pred_bootstrap, 2.5, axis=1
    )

    return y_pred_bootstrap_summary, x_array


def shapiro_wilks(df, y_var, predictions):

    residuals = df[y_var] - predictions
    sw_test = shapiro(residuals)
    print(f"Shapiro-Wilks result: {sw_test}")
    avg_residual = np.mean(residuals)
    print(f"Mean residual result: {avg_residual}")



def heteroscedasticity_test(ols_results):
    white_test = het_white(
        ols_results.resid,
        ols_results.model.exog,
    )
    white_test_results = dict(zip(labels, white_test))
    white_test_pvalue = white_test_results["Test Statistic p-value"]
    if white_test_pvalue < 0.05:
        print(f"heteroscedasticity present! p-value = {white_test_pvalue}")
    else:
        print(f"no heteroscedasticity detected! p-value = {white_test_pvalue}")


def single_scatterplot(tax_credits_df, feature_var, response_var):
    plt.scatter(
        tax_credits_df[feature_var],
        tax_credits_df[response_var],
        alpha=0.005,
        color=demographics_colors[0],
    )
    plt.xlabel(feature_var)
    plt.ylabel(response_var)

def single_ols(tax_credits_df, feature_var, response_var):
    # ols function
    ols_results = ols(tax_credits_df, feature_var, response_var)

    # Test for heteroskedasticity
    #     heteroscedasticity_test(ols_results)

    # bootstrap ols analysis
    (
        y_pred_bootstrap_summary,
        x_array,
    ) = bootstrap_ols_analysis(tax_credits_df, feature_var, response_var)

    # Plot the OLS models with their bootstrapped CIs
    plt.clf()

    pval = round(ols_results.pvalues[1], 2)
    coef = round(ols_results.params[1], 2)

    plt.plot(
        tax_credits_df[feature_var],
        ols_results.fittedvalues,
        demographics_colors[0],
        label=f"OLS Results\nCoef={coef}\nP-val: {pval}",
    )

    plt.fill_between(
        x_array,
        y_pred_bootstrap_summary["95_CI_low"],
        y_pred_bootstrap_summary["95_CI_high"],
        facecolor=demographics_colors[0],
        alpha=0.5,
    )

    plt.axhline(np.mean(tax_credits_df[response_var]), 
                linestyle='--',
                color=demographics_colors[1],
                label="Mean")
    plt.legend()
    plt.show()


def both_response_subplots_ols_with_bootstrap_and_histograms(
    tax_credit_dfs,
    feature_var,
    ylabel,
):
    
    # Run the OLS models
    # Keep track of max y_pred
    # Use the max y_pred to set the y axes matching 

    y_pred_participation_bootstrap_summary_all = []
    x_participation_array_all = []
    participation_pval_all = []
    participation_coef_all = []

    y_pred_value_bootstrap_summary_all = []
    x_value_array_all = []
    value_pval_all = []
    value_coef_all = []

    index_with_xmax = -1
    index_with_participation_max = -1
    index_with_value_max = -1
    running_xmax = 0
    running_participation_max = 0
    running_value_max = 0

    for ind, df in enumerate(tax_credit_dfs):
        
        # Track max response_var
        # Result for participation models
        df_filtered = df[df["percent_returns"].notna() & df["value_returns"].notna()]
        ols_participation_results = ols(df_filtered, feature_var, "percent_returns")
        (
            y_pred_participation_bootstrap_summary,
            x_participation_array,
        ) = bootstrap_ols_analysis(df_filtered, feature_var, "percent_returns")

        participation_pval = format(ols_participation_results.pvalues[1], '.3g') 
        participation_coef = format(ols_participation_results.params[1], '.3g')

        df_filtered_participated = df_filtered[
                df_filtered["Number of returns"] > 0
        ]
        xmin_for_agi = np.min(df_filtered_participated[feature_var])
        xmax_for_agi = np.max(df_filtered_participated[feature_var])
        print(f"AGI class {ind + 1}: xmin = {xmin_for_agi} - xmax = {xmax_for_agi}")
        ols_value_results = ols(df_filtered_participated, feature_var, "value_returns")
        (
            y_pred_value_bootstrap_summary,
            x_value_array,
        ) = bootstrap_ols_analysis(df_filtered_participated, feature_var, "value_returns")

        value_pval = format(ols_value_results.pvalues[1], '.3g') 
        value_coef = format(ols_value_results.params[1], '.3g')

        # Track all the values so we don't have to run them again
        y_pred_participation_bootstrap_summary_all.append(y_pred_participation_bootstrap_summary)
        x_participation_array_all.append(x_participation_array)
        participation_pval_all.append(participation_pval)
        participation_coef_all.append(participation_coef)
        
        y_pred_value_bootstrap_summary_all.append(y_pred_value_bootstrap_summary)
        x_value_array_all.append(x_value_array)
        value_pval_all.append(value_pval)
        value_coef_all.append(value_coef)

        # Calculate the max values to track max y_pred
        current_participation_max = np.max(y_pred_participation_bootstrap_summary["95_CI_high"].values)
        if current_participation_max > running_participation_max:
            running_participation_max = current_participation_max
            index_with_participation_max = ind

        current_value_max = np.max(y_pred_value_bootstrap_summary["95_CI_high"].values)
        if current_value_max > running_value_max:
            running_value_max = current_value_max
            index_with_value_max = ind

        current_xmax = len(df)
        if current_xmax > running_xmax:
            running_xmax = current_xmax
            index_with_xmax = ind

    # Plot the OLS models with their bootstrapped CIs
    plt.clf()
    fig, axes = plt.subplots(
        nrows=3, 
        ncols=6, 
        figsize=(20, 16),
        gridspec_kw={
            # height_ratios[i] / sum(height_ratios)
           'height_ratios': [2, 5, 5],
       'wspace': 0.275,
       'hspace': 0.01}
       )
 
    # axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    hist_labels = ["a)", "b)", "c)", "d)", "e)", "f)"]
    participation_labels = ["g)", "h)", "i)", "j)", "k)", "l)"]
    value_labels = ["m)", "n)", "o)", "p)", "q)", "r)"]
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)

    for ind, df in enumerate(tax_credit_dfs):
        df_filtered = df[df["percent_returns"].notna() & df["value_returns"].notna()]
        df_filtered_participated = df_filtered[
                df_filtered["Number of returns"] > 0
        ]

        # Add histograms
        bins = 20
        colors = ['#4dac26', '#d01c8b']
        axes[0][ind].hist(df_filtered[feature_var], bins=bins,color=colors[0])
        axes[0][ind].hist(df_filtered_participated[feature_var], bins=bins,color=colors[1])
        axes[0][ind].text(0.0, 1.0, hist_labels[ind], transform=axes[0][ind].transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='none', edgecolor='gray', pad=3.0))

        for tick in axes[0][ind].xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

        if ind > 0:
            for tick in axes[0][ind].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

        # # Result for participation models
        ols_participation_results = ols(df_filtered, feature_var, "percent_returns")

        # Plot the participation data
        axes[1][ind].plot(
            df_filtered[feature_var],
            ols_participation_results.fittedvalues,
            demographics_colors[0],
            label=f"Coef={participation_coef_all[ind]}\nP-val: {participation_pval_all[ind]}",
        )

        axes[1][ind].fill_between(
            x_participation_array_all[ind],
            y_pred_participation_bootstrap_summary_all[ind]["95_CI_low"],
            y_pred_participation_bootstrap_summary_all[ind]["95_CI_high"],
            facecolor=demographics_colors[0],
            alpha=0.5,
        )
        axes[1][ind].legend(loc='upper right')

        axes[1][ind].axhline(np.mean(df_filtered["percent_returns"]), 
                          linestyle='--', 
                          color=demographics_colors[1],
                          label="Mean")
        axes[1][ind].text(0.0, 1.0, participation_labels[ind], transform=axes[1][ind].transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='none', edgecolor='gray', pad=3.0))

        for tick in axes[1][ind].xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        
        if ind > 0:
            for tick in axes[1][ind].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)


        # Result for value models
        # only use those that participated when evaluating average value
        df_filtered_participated = df_filtered[
                df_filtered["Number of returns"] > 0
        ]
        ols_value_results = ols(df_filtered_participated, feature_var, "value_returns")

        # Plot the value data
        axes[2][ind].plot(
            df_filtered_participated[feature_var],
            ols_value_results.fittedvalues,
            demographics_colors[0],
            label=f"Coef={value_coef_all[ind]}\nP-val: {value_pval_all[ind]}",
        )

        axes[2][ind].fill_between(
            x_value_array_all[ind],
            y_pred_value_bootstrap_summary_all[ind]["95_CI_low"],
            y_pred_value_bootstrap_summary_all[ind]["95_CI_high"],
            facecolor=demographics_colors[0],
            alpha=0.5,
        )
        axes[2][ind].legend(loc='upper right')

        axes[2][ind].axhline(np.mean(df_filtered_participated["value_returns"]), 
                          linestyle='--', 
                          color=demographics_colors[1],
                          label="Mean")

        axes[2][ind].text(0.0, 1.0, participation_labels[ind], transform=axes[2][ind].transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='none', edgecolor='gray', pad=3.0))
        
        if ind > 0:
            for tick in axes[2][ind].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

        # Line up axis labels:
        axes[1][ind].set_xlim(left=0, right=100)
        axes[0][ind].sharex(axes[1][ind])
        axes[2][ind].sharex(axes[1][ind])


        axes[0][ind].sharey(axes[0][index_with_xmax])
        axes[1][ind].sharey(axes[1][index_with_participation_max])
        axes[2][ind].sharey(axes[2][index_with_value_max])


        if feature_var == "percent_white":

            axes[2][ind].set_xlabel(f"AGI Stub: {ind+1}\nPercent White Individuals")

        elif feature_var == "percent_homeowners":
            axes[2][ind].set_xlabel(f"AGI Stub: {ind+1}\nPercent Homeowners")

        elif (feature_var == "log_population_density") or (feature_var == "population_density"):
            axes[2][ind].set_xlabel(f"AGI Stub: {ind+1}\nPopulation Density")

    axes[0][0].set_ylabel("Number of \n Zip Codes")
    axes[1][0].set_ylabel("Percent of Tax Returns \nwith Energy Efficiency Tax Credit")
    axes[2][0].set_ylabel("Average Value of Energy Efficiency\n Tax Credit Received")
    
    plt.savefig(
        os.path.join(FIGURES_DIR, f"{feature_var}_both_responses_linear.pdf"),
        format="pdf",
        bbox_inches="tight",
    )

    plt.show()


def poly_both_response_subplots_ols_with_bootstrap_and_histograms(
    tax_credit_dfs,
    feature_var,
):
    
    # Plot the OLS models with their bootstrapped CIs
    plt.clf()
    fig, axes = plt.subplots(
        nrows=3, 
        ncols=6, 
        figsize=(20, 16),
        gridspec_kw={
            # height_ratios[i] / sum(height_ratios)
           'height_ratios': [2, 5, 5],
       'wspace': 0.275,
       'hspace': 0.01}
       )

    # Run the OLS models
    # Keep track of max y_pred
    # Use the max y_pred to set the y axes matching 

    y_pred_participation_bootstrap_summary_all = []
    x_participation_array_all = []
    participation_pval_all = []
    participation_coef_all = []

    y_pred_value_bootstrap_summary_all = []
    x_value_array_all = []
    value_pval_all = []
    value_coef_all = []

    index_with_xmax = -1
    index_with_participation_max = -1
    index_with_value_max = -1
    running_xmax = 0
    running_participation_max = 0
    running_value_max = 0

    for ind, df in enumerate(tax_credit_dfs):
        print(f"Analysis of bucket {ind+1}")
        # Track max response_var
        # Result for participation models
        df_filtered = df[df["percent_returns"].notna() & df["value_returns"].notna()]
        kfold_rss_results_df = poly_order_cv(
            df_filtered, feature_var, "percent_returns",
            k,
            poly_order_max,
        )

        poly_order = one_standard_error_rule(kfold_rss_results_df)
        poly_features = PolynomialFeatures(degree=poly_order)
        X_polyfeatures = poly_features.fit_transform(
                df_filtered[feature_var].values.reshape(-1,1)
            )
        y = df_filtered["percent_returns"]
        poly_features_model = sm.OLS(y, X_polyfeatures)
        ols_participation_results = poly_features_model.fit()
        #ols_participation_results = ols(df_filtered, feature_var, "percent_returns")
        (
            y_pred_participation_bootstrap_summary,
            x_participation_array,
        ) = bootstrap_poly_ols_analysis(df_filtered, feature_var, "percent_returns", poly_order)

        participation_pval = format(ols_participation_results.pvalues[1], '.3g') 
        participation_coef = format(ols_participation_results.params[1], '.3g')

        df_filtered_participated = df_filtered[
                df_filtered["Number of returns"] > 0
        ]
        xmin_for_agi = np.min(df_filtered_participated[feature_var])
        xmax_for_agi = np.max(df_filtered_participated[feature_var])
        # print(f"AGI class {ind + 1}: xmin = {xmin_for_agi} - xmax = {xmax_for_agi}")

        kfold_rss_results_df = poly_order_cv(
            df_filtered_participated, feature_var, "value_returns",
            k,
            poly_order_max,
        )

        poly_order = one_standard_error_rule(kfold_rss_results_df)
        poly_features = PolynomialFeatures(degree=poly_order)
        X_polyfeatures = poly_features.fit_transform(
                df_filtered_participated[feature_var].values.reshape(-1,1)
            )
        y = df_filtered_participated["value_returns"]
        poly_features_model = sm.OLS(y, X_polyfeatures)
        ols_value_results = poly_features_model.fit()

        # ols_value_results = ols(df_filtered_participated, feature_var, "value_returns")
        (
            y_pred_value_bootstrap_summary,
            x_value_array,
        ) = bootstrap_poly_ols_analysis(df_filtered_participated, feature_var, "value_returns", poly_order)

        value_pval = format(ols_value_results.pvalues[1], '.3g') 
        value_coef = format(ols_value_results.params[1], '.3g')

        # Track all the values so we don't have to run them again
        y_pred_participation_bootstrap_summary_all.append(y_pred_participation_bootstrap_summary)
        x_participation_array_all.append(x_participation_array)
        participation_pval_all.append(participation_pval)
        participation_coef_all.append(participation_coef)
        
        y_pred_value_bootstrap_summary_all.append(y_pred_value_bootstrap_summary)
        x_value_array_all.append(x_value_array)
        value_pval_all.append(value_pval)
        value_coef_all.append(value_coef)

        # Calculate the max values to track max y_pred
        current_participation_max = np.max(y_pred_participation_bootstrap_summary["95_CI_high"].values)
        if current_participation_max > running_participation_max:
            running_participation_max = current_participation_max
            index_with_participation_max = ind

        current_value_max = np.max(y_pred_value_bootstrap_summary["95_CI_high"].values)
        if current_value_max > running_value_max:
            running_value_max = current_value_max
            index_with_value_max = ind

        current_xmax = len(df)
        if current_xmax > running_xmax:
            running_xmax = current_xmax
            index_with_xmax = ind

    
 
    # axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    hist_labels = ["a)", "b)", "c)", "d)", "e)", "f)"]
    participation_labels = ["g)", "h)", "i)", "j)", "k)", "l)"]
    value_labels = ["m)", "n)", "o)", "p)", "q)", "r)"]
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)

    for ind, df in enumerate(tax_credit_dfs):
        df_filtered = df[df["percent_returns"].notna() & df["value_returns"].notna()]
        df_filtered_participated = df_filtered[
                df_filtered["Number of returns"] > 0
        ]

        # Add histograms
        bins = 20
        colors = ['#4dac26', '#d01c8b']
        axes[0][ind].hist(df_filtered[feature_var], bins=bins,color=colors[0])
        axes[0][ind].hist(df_filtered_participated[feature_var], bins=bins,color=colors[1])
        axes[0][ind].text(0.0, 1.0, hist_labels[ind], transform=axes[0][ind].transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='none', edgecolor='gray', pad=3.0))

        for tick in axes[0][ind].xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

        if ind > 0:
            for tick in axes[0][ind].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

        # # Result for participation models
        kfold_rss_results_df = poly_order_cv(
            df_filtered, feature_var, "percent_returns",
            k,
            poly_order_max,
        )
        poly_order = one_standard_error_rule(kfold_rss_results_df)
        poly_features = PolynomialFeatures(degree=poly_order)
        df_filtered = df_filtered.sort_values(by=feature_var)
        X_polyfeatures = poly_features.fit_transform(
                df_filtered[feature_var].values.reshape(-1,1)
            )
        y = df_filtered["percent_returns"]
        poly_features_model = sm.OLS(y, X_polyfeatures)
        ols_participation_results = poly_features_model.fit()

        #ols_participation_results = ols(df_filtered, feature_var, "percent_returns")

        # Plot the participation data
        axes[1][ind].plot(
            df_filtered[feature_var],
            ols_participation_results.fittedvalues,
            demographics_colors[0],
            label=f"Coef={participation_coef_all[ind]}\nP-val: {participation_pval_all[ind]}",
        )

        axes[1][ind].fill_between(
            x_participation_array_all[ind],
            y_pred_participation_bootstrap_summary_all[ind]["95_CI_low"],
            y_pred_participation_bootstrap_summary_all[ind]["95_CI_high"],
            facecolor=demographics_colors[0],
            alpha=0.5,
        )
        axes[1][ind].legend(loc='upper right')

        axes[1][ind].axhline(np.mean(df_filtered["percent_returns"]), 
                          linestyle='--', 
                          color=demographics_colors[1],
                          label="Mean")
        axes[1][ind].text(0.0, 1.0, participation_labels[ind], transform=axes[1][ind].transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='none', edgecolor='gray', pad=3.0))

        for tick in axes[1][ind].xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        
        if ind > 0:
            for tick in axes[1][ind].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)


        # Result for value models
        # only use those that participated when evaluating average value
        df_filtered_participated = df_filtered[
                df_filtered["Number of returns"] > 0
        ]
        kfold_rss_results_df = poly_order_cv(
            df_filtered_participated,
            feature_var, "value_returns",
            k,
            poly_order_max,
        )

        poly_order = one_standard_error_rule(kfold_rss_results_df)
        poly_features = PolynomialFeatures(degree=poly_order)
        df_filtered_participated = df_filtered_participated.sort_values(by=feature_var)
        X_polyfeatures = poly_features.fit_transform(
                df_filtered_participated[feature_var].values.reshape(-1,1)
            )
        y = df_filtered_participated["value_returns"]
        poly_features_model = sm.OLS(y, X_polyfeatures)
        ols_value_results = poly_features_model.fit()
        #ols_value_results = ols(df_filtered_participated, feature_var, "value_returns")

        # Plot the value data
        axes[2][ind].plot(
            df_filtered_participated[feature_var],
            ols_value_results.fittedvalues,
            demographics_colors[0],
            label=f"Coef={value_coef_all[ind]}\nP-val: {value_pval_all[ind]}",
        )

        axes[2][ind].fill_between(
            x_value_array_all[ind],
            y_pred_value_bootstrap_summary_all[ind]["95_CI_low"],
            y_pred_value_bootstrap_summary_all[ind]["95_CI_high"],
            facecolor=demographics_colors[0],
            alpha=0.5,
        )
        axes[2][ind].legend(loc='upper right')

        axes[2][ind].axhline(np.mean(df_filtered_participated["value_returns"]), 
                          linestyle='--', 
                          color=demographics_colors[1],
                          label="Mean")

        axes[2][ind].text(0.0, 1.0, participation_labels[ind], transform=axes[2][ind].transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='none', edgecolor='gray', pad=3.0))
        
        if ind > 0:
            for tick in axes[2][ind].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

        # Line up axis labels:
        axes[1][ind].set_xlim(left=0, right=100)
        axes[0][ind].sharex(axes[1][ind])
        axes[2][ind].sharex(axes[1][ind])


        axes[0][ind].sharey(axes[0][index_with_xmax])
        axes[1][ind].sharey(axes[1][index_with_participation_max])
        axes[2][ind].sharey(axes[2][index_with_value_max])


        if feature_var == "percent_white_alone":

            axes[2][ind].set_xlabel(f"Median Income \ncorresponsing to \nAGI class {ind+1}\nPercent White Individuals")

        elif feature_var == "percent_homeowners":
            axes[2][ind].set_xlabel(f"AGI class: {ind+1}\nPercent Homeowners")

        elif (feature_var == "log_population_density") or (feature_var == "population_density"):
            axes[2][ind].set_xlabel(f"AGI class: {ind+1}\nPopulation Density")

    axes[0][0].set_ylabel("Number of \n Zip Codes")
    axes[1][0].set_ylabel("Percent of Tax Returns \nwith Energy Efficiency Tax Credit")
    axes[2][0].set_ylabel("Average Value of Energy Efficiency\n Tax Credit Received")
    
    plt.savefig(
        os.path.join(FIGURES_DIR, f"poly_{feature_var}_both_responses_linear.pdf"),
        format="pdf",
        bbox_inches="tight",
    )

    plt.show()


##################################################
# With Quantile Regression instead of Polynomial #
##################################################


def poly_order_cv_quantile_regression(df, feature_var, response_var, k, poly_order_max):

    kf = KFold(n_splits=k, shuffle=True, random_state=1234)
    kfold_rss_results_df = pd.DataFrame(columns=[1, 2, 3, 4, 5])
    fold = 1

    for train_index, val_index in kf.split(df[feature_var]):
        # print(f"Fold #{fold}")
        df_fold_train = df.iloc[train_index]
        df_fold_val = df.iloc[val_index]

        rss_fold_results = []

        for poly_order in range(1, poly_order_max + 1):

            mod = smf.quantreg(f"{response_var} ~ {feature_var} + I({feature_var} ** {poly_order})", df_fold_train)

            # Quantile regression for 4 quantiles

            quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]

            # get all result instances in a list
            # res_all = [mod.fit(q=q) for q in quantiles]
            res_50 = mod.fit(q=0.50)
            
            # create x for prediction
            # x_p = np.linspace(
            #     df_fold_val[feature_var].min(),
            #     df_fold_val[feature_var].max(),
            #     50,
            # )
            x_p = df_fold_val[feature_var].values

            res_50_preds = res_50.predict({feature_var: x_p})
            #print(f"PREDICTIONS = {res_50_preds}")
            residuals = res_50_preds - df_fold_val[response_var].values
            #print(f"residuals = {residuals}")
            residuals_squared = [i**2 for i in residuals]
            RSS_val = np.sum(residuals_squared)
            #print(f"RSS_val = {RSS_val}")
            # Track RSS results per polynomial feature per fold
            rss_fold_results.append(RSS_val)

        kfold_rss_results_df[fold] = rss_fold_results
        fold = fold + 1

    return kfold_rss_results_df


# Fit an OLS model to a single Bootstrap sample
def bootstrap_quantile_regression_sample_analysis(
    df_bootstrap, feature_var, response_var, x_array, polynomial
):
    x_bootstrap_min = df_bootstrap[feature_var].min()
    x_bootstrap_max = df_bootstrap[feature_var].max()

    mod = smf.quantreg(f"{response_var} ~ {feature_var} + I({feature_var} ** {polynomial})", df_bootstrap)

    # Quantile regression for 4 quantiles

    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]

    # get all result instances in a list
    # res_all = [mod.fit(q=q) for q in quantiles]
    res_50 = mod.fit(q=0.50)
    

    # You do no want to extrapolate the bootstrapped regression beyond the range of the bootstrap data
    # For values below the lowest income in the bootstrap sample and above the highest income in the
    # bootstrap sample, place NaN values.
    entries_below_bootstrap_min = sum(x_array < x_bootstrap_min)
    y_pred_below_bootstrap_min = np.full(entries_below_bootstrap_min, np.nan)

    entries_above_bootstrap_max = sum(x_array > x_bootstrap_max)
    y_pred_above_bootstrap_max = np.full(entries_above_bootstrap_max, np.nan)

    # For entries within the bootstrap sample range, used your fitted regression to predict
    values_within_bootstrap_data = x_array[
        entries_below_bootstrap_min : (
            len(x_array) - entries_above_bootstrap_max
        )
    ]
    #values_within_bootstrap_data = values_within_bootstrap_data.reshape(-1, 1)
    #values_within_bootstrap_data_polyfeatures = poly_features.fit_transform(values_within_bootstrap_data)
    # y_pred_within_bootstrap_data_range = res_50.predict(
    #     values_within_bootstrap_data
    # )
    y_pred_within_bootstrap_data_range = res_50.predict({feature_var: values_within_bootstrap_data})

    # Concatenate the preds all three regions
    y_pred = np.concatenate(
        (
            y_pred_below_bootstrap_min,
            y_pred_within_bootstrap_data_range,
            y_pred_above_bootstrap_max,
        )
    )

    return y_pred


# Bootstrap function to get confidence interval for OLS models
def bootstrap_poly_quantile_regression_analysis(df, variable_of_interest, response_of_interest, polynomial):
    X = df[variable_of_interest]
    X = X.reset_index(drop=True)
    y = df[response_of_interest]
    y = y.reset_index(drop=True)

    x_array = np.linspace(df[variable_of_interest].min(),df[variable_of_interest].max(),100)

    y_pred_bootstrap = pd.DataFrame()
    for i in range(0, 1000):
        df_bootstrap = df.sample(n=len(df), replace=True)
        
        y_pred = bootstrap_quantile_regression_sample_analysis(
            df_bootstrap, variable_of_interest, response_of_interest, x_array, polynomial
        )
        y_pred_bootstrap[i] = y_pred

    y_pred_bootstrap_summary = pd.DataFrame()
    y_pred_bootstrap_summary["median"] = np.nanpercentile(
        y_pred_bootstrap, 50, axis=1
    )
    y_pred_bootstrap_summary["95_CI_high"] = np.nanpercentile(
        y_pred_bootstrap, 97.5, axis=1
    )
    y_pred_bootstrap_summary["95_CI_low"] = np.nanpercentile(
        y_pred_bootstrap, 2.5, axis=1
    )

    return y_pred_bootstrap_summary, x_array


def run_quantile_regression(df, x_var, y_var, skip1=False): 
    #mod = smf.quantreg(f"{y_var} ~ {x_var}+ I({x_var} ** 1.0)", df)
    mod = smf.quantreg(f"{y_var} ~ {x_var}", df)
    # Quantile regression for 4 quantiles
    quantiles = [0.25, 0.50, 0.75]
    # get all result instances in a list
    res_all = []
    for q in quantiles:
        res = mod.fit(q=q, missing="drop")
        if q == 0.5:
            print(res.summary())
            conf_int_low = res.conf_int(alpha=0.05, cols=None).iloc[1,0]
            conf_int_high = res.conf_int(alpha=0.05, cols=None).iloc[1,1]
            print(f"{res.params[1]}, {res.pvalues[1]}, {conf_int_low}, {conf_int_high} ")
            print("\n\n")
        res_all.append(res)

    #create x for prediction
    x_p = np.linspace(
        df[x_var].min(),
        df[x_var].max(),
        50,
    )
    #x_p = df[x_var]
    df_p = pd.DataFrame({x_var: x_p})

    for qm, res in zip(quantiles, res_all):
        df_p[qm] = res.predict({x_var: x_p})

    return df_p

def run_quantile_regression_with_cluster_standard_errors(df, x_var, y_var, skip1=False): 
    from pyqreg import quantreg
    mod = quantreg(f"{y_var} ~ {x_var}", df)
    # Quantile regression for 4 quantiles
    quantiles = [0.25, 0.50, 0.75]
    # get all result instances in a list
    res_all = []
    for q in quantiles:
        res = mod.fit(q=q, cov_type='cluster')
        if q == 0.5:
            print(res.summary())
        res_all.append(res)

    #create x for prediction
    x_p = np.linspace(
        df[x_var].min(),
        df[x_var].max(),
        50,
    )
    #x_p = df[x_var]
    df_p = pd.DataFrame({x_var: x_p})

    for qm, res in zip(quantiles, res_all):
        df_p[qm] = res.predict({x_var: x_p})

    return df_p

def round_down(num, divisor):
    return num - (num % divisor)


def calculate_bars_density(X):
    df = pd.DataFrame(columns=["X"])
    df["X"] = X
    x_bars = np.arange(0, 100, 5)
    y_bars = []
    for x_bar in x_bars:
        min_x = x_bar
        max_x = x_bar + 5
        df_bar = df[(df["X"] >= min_x) & (df["X"] < max_x)]
        count = len(df_bar)
        y_bars.append(np.log(count))
        # y_bars.append(count)
    return x_bars, y_bars

def quantile_regression_both_response_subplots_with_bootstrap_and_histograms_updated(
    tax_credit_dfs,
    feature_var,
    participation_poly_orders,
    value_poly_orders,
    segment="urban"
):
    
    # Plot the models with their bootstrapped CIs
    # plt.clf()
    fig, axes = plt.subplots(
        nrows=3, 
        ncols=6, 
        figsize=(20, 16),
        gridspec_kw={
            # height_ratios[i] / sum(height_ratios)
           'height_ratios': [2, 5, 5],
       'wspace': 0.275,
       'hspace': 0.05}
       )

    # Run the OLS models
    # Keep track of max y_pred
    # Use the max y_pred to set the y axes matching 
    df_p_participation_all = []
    df_p_value_all = []

    y_preds_value_all = []
    y_pred_value_bootstrap_summary_all = []
    x_value_array_all = []

    index_with_xmax = -1
    index_with_participation_max = -1
    index_with_value_max = -1
    running_xmax = 0
    running_participation_max = 0
    running_value_max = 0

    hist_labels = ["a)", "b)", "c)", "d)", "e)", "f)"]
    participation_labels = ["g)", "h)", "i)", "j)", "k)", "l)"]
    value_labels = ["m)", "n)", "o)", "p)", "q)", "r)"]
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)

    for ind, df in enumerate(tax_credit_dfs):

        current_xmax = len(df)
        if current_xmax > running_xmax:
            running_xmax = current_xmax
            index_with_xmax = ind

        print(f"Analysis of bucket {ind+1}")
        df_filtered = df[df["percent_returns"].notna() & df["value_returns"].notna()].sort_values(by="percent_returns")
        # df_filtered = df.copy().sort_values(by="percent_returns")
        df_filtered_participated = df_filtered[
                df_filtered["Number of returns"] > 0
        ].sort_values(by="value_returns")

        print(f"analyzing participation")
        df_p_participation = run_quantile_regression(df_filtered, feature_var, "percent_returns")
        print(f"mean = {df_filtered.percent_returns.mean()}")
        print(f"median = {df_filtered.percent_returns.median()}")
        df_p_participation_all.append(df_p_participation)

        current_participation_max = np.max(df_p_participation[0.75].values)
        if current_participation_max > running_participation_max:
            running_participation_max = current_participation_max
            index_with_participation_max = ind

        print(f"analyzing value")
        print(f"mean = {df_filtered_participated.value_returns.mean()}")
        print(f"median = {df_filtered_participated.value_returns.median()}")
        df_p_value = run_quantile_regression(df_filtered_participated, feature_var, "value_returns")
        df_p_value_all.append(df_p_value)

        # Calculate the max values to track max y_pred
        current_value_max = np.max(df_p_value[0.75].values)
        if current_value_max > running_value_max:
            running_value_max = current_value_max
            index_with_value_max = ind

    for ind, df in enumerate(tax_credit_dfs):
        # Add histograms
        bins = 20
        colors = ['#4dac26', '#d01c8b', "#2c7bb6", "#fdae61"]
        # axes[0][ind].hist(df_filtered[feature_var], bins=bins,color=colors[0])
        # axes[0][ind].hist(df_filtered_participated[feature_var], bins=bins,color=colors[1])
        df_filtered = df[df["percent_returns"].notna() & df["value_returns"].notna()].sort_values(by="percent_returns")
        df_filtered_participated = df_filtered[
                df_filtered["Number of returns"] > 0
        ].sort_values(by="value_returns")
        x_bars, y_bars = calculate_bars_density(df_filtered[feature_var])

        x_bars_participated, y_bars_participated = calculate_bars_density(
            df_filtered_participated[feature_var]
        )

        axes[0][ind].bar(
            x_bars, y_bars, width=4, color=colors[0]
        )
        axes[0][ind].bar(
            x_bars_participated,
            y_bars_participated,
            color=colors[1],
            width=4,
        )
        axes[0][ind].text(0.0, 1.0, hist_labels[ind], transform=axes[0][ind].transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='none', edgecolor='gray', pad=3.0))


        for tick in axes[0][ind].xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

        if ind > 0:
            for tick in axes[0][ind].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)


        # Plot the participation data
        quantiles = [0.25, 0.50, 0.75]
        for qm in quantiles:
            if qm != 0.50:
                axes[1][ind].plot(
                    df_p_participation_all[ind][feature_var],
                    df_p_participation_all[ind][qm],
                    colors[2],
                    linestyle="dotted",
                    #alpha=0.75
                )
            else:
                axes[1][ind].plot(
                    df_p_participation_all[ind][feature_var],
                    df_p_participation_all[ind][qm],
                    colors[2],
                )

        axes[1][ind].axhline(np.mean(df_filtered["percent_returns"]), 
                          linestyle='--', 
                          color=colors[3],
                          label="Mean")
        axes[1][ind].text(0.0, 1.0, participation_labels[ind], transform=axes[1][ind].transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='none', edgecolor='gray', pad=3.0))

        for tick in axes[1][ind].xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        
        if ind > 0:
            for tick in axes[1][ind].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

        for qm in quantiles:
            if qm != 0.50:
                axes[2][ind].plot(
                    df_p_value_all[ind][feature_var],
                    df_p_value_all[ind][qm],
                    colors[2],
                    linestyle="dotted",
                    #alpha=0.75
                )
            else:
                axes[2][ind].plot(
                    df_p_value_all[ind][feature_var],
                    df_p_value_all[ind][qm],
                    colors[2],
                )
        
        axes[2][ind].axhline(np.mean(df_filtered_participated["value_returns"]), 
                          linestyle='--', 
                          color=colors[3],
                          label="Mean")

        axes[2][ind].text(0.0, 1.0, value_labels[ind], transform=axes[2][ind].transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='none', edgecolor='gray', pad=3.0))
        
        if ind > 0:
            for tick in axes[2][ind].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)


        # Line up axis labels:
        axes[1][ind].set_xlim(left=0, right=100)
        axes[0][ind].sharex(axes[1][ind])
        axes[2][ind].sharex(axes[1][ind])


        axes[0][ind].sharey(axes[0][index_with_xmax])
        axes[1][ind].sharey(axes[1][index_with_participation_max])
        axes[2][ind].sharey(axes[2][index_with_value_max])

        # axes[0][ind].set_ylim(bottom=-2)
        # axes[1][ind].set_ylim(bottom=-0.4)
        # axes[2][ind].set_ylim(bottom=-2)

        if feature_var == "percent_white_alone":

            axes[2][ind].set_xlabel(f"Percent White Individuals\n Median Income \ncorresponsing to \nAGI class {ind+1}\n")

        elif feature_var == "percent_homeowners":
            axes[2][ind].set_xlabel(f"Percent Owner-\noccupied Households\nAGI class: {ind+1}")

        elif (feature_var == "log_population_density") or (feature_var == "population_density"):
            axes[2][ind].set_xlabel(f"AGI class: {ind+1}\nPopulation Density")

    axes[0][0].set_ylabel("Number of \n Zip Codes\n (log scale)")
    axes[1][0].set_ylabel("Percent of Tax Returns \nwith RETC")
    axes[2][0].set_ylabel("Average Value \nReceived from RETC")
    
    # plt.gcf()
    plt.savefig(
        os.path.join(FIGURES_DIR, f"quantiles_only_{feature_var}_linear_{segment}_cse.pdf"),
        format="pdf",
        bbox_inches="tight",
    )

    #plt.show()
    return
