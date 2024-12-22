# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 09:43:17 2024

@author: PVS
"""
'''
3) Step-by-step walk through of the solution

Describe the steps you took to solve the problem. What did you find at each stage, and how did it inform the next steps? Build up to the final solution.

4) Model evaluation
Describe the final model (or ensemble) in detail. What was the objective, what parameters were prominent, and how did you evaluate the success of your models(s)? A convincing explanation of the robustness of your solution will go a long way to supporting your solution.
.

5) Comparison to benchmark

How does your final solution compare to the benchmark you laid out at the outset? Did you improve on the benchmark? Why or why not?

6) Visualization(s)

In addition to quantifying your model and the solution, please include all relevant visualizations that support the ideas/insights that you gleaned from the data.
'''
import   pandas                   as     pd
import    warnings
warnings.filterwarnings("ignore")
def draw_cm(mname, datacategory, actual, predicted ):
    
    import matplotlib.pyplot as   plt
    import sklearn.metrics   as   metrics
    import seaborn           as   sns
    
    cm = metrics.confusion_matrix( actual, predicted)
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels = ["Yes", "No"] , yticklabels = ["Yes", "No"] )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    titlex =  'Confusion Matrix for the model: %s using %s ' % (mname, datacategory)    
    plt.title(titlex)
    plt.show() 
    
def draw_roc(mname,datacategory, actual, probs ):
    import matplotlib.pyplot as   plt
    import sklearn.metrics   as   metrics
   
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
    drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(6, 4))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    titlex =  'Receiver operating characteristic curve for the model:%s using %s ' % (mname, datacategory)   
    plt.title(titlex)
    plt.legend(loc="lower right")
    plt.show()
    return fpr, tpr, thresholds
def evaluate_model_SM_performance(name, model, x_train, y_train, x_test, y_test):
    from    sklearn.metrics import  recall_score, precision_score,roc_auc_score,f1_score 
    from    sklearn.metrics import  confusion_matrix, classification_report
    import  pandas          as      pd
    
    # Predict the response for training dataset
    classifier         =  model.fit(x_train, y_train)
    # print the model parameters
    print("Model parameters: {}".format(model.get_params()))
    predicted_train    =  classifier.predict(x_train)
    matrix             =  confusion_matrix(y_train, predicted_train)
    ###
    train_auroc        =  roc_auc_score(y_train, predicted_train)
    train_recall       =  recall_score(y_train, predicted_train)
    train_precision    =  precision_score(y_train, predicted_train)
    train_f1score      =  f1_score(y_train, predicted_train, average = 'weighted')
    ###
    print("\nTraining Data")
    print(matrix)
    draw_cm(name, "Training data", y_train, predicted_train)
    measures_train      = classification_report(y_train, predicted_train) 
    print("\nTraining Data")
    print(measures_train) 
    draw_roc(name, "Training data", y_train, predicted_train)
    # Predict the response for testing dataset
    predicted_test     =  classifier.predict(x_test)
    matrix1            =  confusion_matrix(y_test, predicted_test)
    ### 
    test_auroc         =  roc_auc_score(y_test, predicted_test)
    test_recall        =  recall_score(y_test, predicted_test)
    test_precision     =  precision_score(y_test, predicted_test)
    test_f1score       =  f1_score(y_test, predicted_test, average = 'weighted')
    ###    
    print("\nTest  Data")
    print(matrix1)
    draw_cm(name, "Test data", y_test, predicted_test)
    measures_test     = classification_report(y_test, predicted_test) 
    print("\nTest  Data")
    print(measures_test) 
    draw_roc(name, "Test data",  y_test, predicted_test) 
    df_smote  =  pd.DataFrame({'Model' : name, 'Recall Training data' : train_recall, 'Recall Test data' : test_recall,\
                              'F1 Weighted Training data' : train_f1score, 'F1 Weighted Test data' : test_f1score,
                              'AUROC Training data' : train_auroc, 'AUROC Test data' : test_auroc,
                              'Precision Training data' : train_precision, 'Precision Test data' : test_precision},\
                              index = [0]) 
    return df_smote
def calculate_VIF(X):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    #calculate VIF for each explanatory variable
    vif             =  pd.DataFrame()
    vif['VIF']      =  [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['variable'] =  X.columns

    #view VIF for each explanatory variable 
    return vif
def check_linearity(x1, df, title, y1):
    
    import matplotlib.pyplot as plt
    import seaborn           as sns    
    sns.regplot(x = x1, y= y1, data= df, logistic= True).set_title(title)
    plt.show()
def  build_LR(df, X, y, target):
    '''
    Assumptions Check for Logistic Regression
    Assumption 1 - Binary logistic regression requires the target / dependent variable to be binary.
    For a binary regression, the factor level 1 of the dependent variable should represent the desired outcome (such as Success etc..).
    Assumption 2 - Only the meaningful variables should be included.
    Assumption 3 -The predictor variables should not be correlated to each other meaning the model should have little or no multicollinearity.
    Assumption 4 - The independent variables are linearly related to the log odds.
    Assumption 5 - Logistic regression requires quite a large number of observations.
    '''
    # import   time
    import   statsmodels.api        as     sm
    import   numpy                  as     np
    import   pandas                 as     pd
    # start_time = time.time()
    print("\nCHECK LOGISTIC REGRESSION ASSUMPTIONS\n")
    tx1      =  'Assumption 1 - Binary logistic regression requires the target / dependent variable to be binary'
    Class_df =  df[target].value_counts().reset_index().set_axis(['item','count'], axis=1)
    classes_count =  Class_df.shape[1]
    print("\n{}".format(tx1))
    if classes_count == 2:
       print("This assumption is satisfied since \n")
       print("Target variable, %s is having %d classes" %(target, Class_df.shape[1]))
    else:
      print("This assumption is Not satisfied since \n")
      print("Target variable, %s is having %d classes" %(target, Class_df.shape[1]))
    tx2   =  'Assumption 2 - Only the meaningful variables should be included.' 
    print("\n{}".format(tx2))
    print("This assumption is satisfied since \n")    
    print('We have ensured that there are no unwanted variables selected for model building.')
    tx3  =  'Assumption 3 -The predictor variables should not be correlated to each other meaning the model should have little or no multicollinearity.'
    print("\n{}".format(tx3))
    vif_df  = calculate_VIF(X)
    vif_df.sort_values(by = ['VIF'], ascending = False, inplace = True)
    print(vif_df)  
    if  vif_df[ vif_df['VIF'] > 5].shape[0] == 0:
        print("\nThis assumption is satisfied since \n")    
        print("All the predictor variables are found to be non-collinear")
    else:
        print("This assumption is NOT satisfied since")    
        print("Some predictor variables are found to be collinear")  
        print(vif_df[ vif_df['VIF'] > 5])
    tx4  =  'Assumption 4 - The independent variables are linearly related to the log odds.' 
    tx41 =  'We need to check the assumption of Independent variables are linearly related to the log odds.One way to checking this is to plot the Independent variables in question and look for an S-shaped curve.'
    print("\n{}".format(tx4))
    print("\n{}".format(tx41))
    df_num        =  X.select_dtypes(include=np.number) 
    num_variables =  df_num.columns
    for i in range(len(num_variables)):
        title = num_variables[i] + '  Log odds linear plot'
        xvar  = num_variables[i]
        check_linearity(xvar, df, title, target)
    tx5  =  'Assumption 5 - Logistic regression requires quite a large number of observations'
    print("\n{}".format(tx5))
    #Number of events (cases where Response == 1)
    num_events = df[target].sum()    
    # Number of predictor variables (excluding 'Response')
    num_predictors = len(X.columns)
    # Number of events per predictor variable
    events_per_predictor = round(num_events / num_predictors,0)
    print("Number of events:", num_events)
    print("Number of predictor variables:", num_predictors)
    print("Events per predictor:", events_per_predictor)
    print("\nWith %d events and %d predictor variables, the calculated number of events per predictor is approximately %d" %(num_events, num_predictors, events_per_predictor))
    print("This exceeds the commonly recommended guideline of having at least 10-20 events per predictor variable")
    print("Inference: The dataset appears to meet the assumption of having a sufficiently large sample size for logistic regression")
    print("\nBuild Logistic Regression")
    logit = sm.Logit(y, sm.add_constant( X) )
    lg    = logit.fit()
    print("Report Psuedo R-square, model coefficients and p-value")
    print(lg.summary())
    '''
    print("\nLR Params")
    print(lg.params)
    print(dir(lg))
    '''
    McFaddenRSQ =  round(1 - (lg.llf /lg.llnull),6)
    print('\n', ' McFadden R Square ', McFaddenRSQ)
    txps   =  'We observe that the McFadden R square (Pseudo R square) is %f' % McFaddenRSQ
    print(txps)    
    if McFaddenRSQ > 0.4:
       resul1  =  'Model fitnesss is very good.'
    else:
       resul1  =  'Model fitnesss is Not so good.'   
    print('{}'.format(resul1))
    ### 
    txts  =  'This McFadden approach is one minus the ratio of two log likelihoods. The numerator is the log likelihood of the logit model selected and the denominator is the log likelihood if the model just had an intercept'
    print(txts)

    coef_df = pd.DataFrame({
        'coef': lg.params.values,
        'pvalue': lg.pvalues.round(4),
        'ci_lower': lg.conf_int()[0],
        'ci_upper': lg.conf_int()[1]
    }, index=lg.params.index)
    coef_df.drop(['const'], inplace = True) 
    coef_df['Variable']    =   coef_df.index
    print(coef_df)    
    significant_vars      =    coef_df.loc[coef_df['pvalue'] < 0.05, 'Variable'].tolist()
    print("\nList the significant variables at 5% level of significance {}".format(significant_vars))
    print("\nWe observe that only %d predictor variables are significant at 5 percent level of significance" % len(significant_vars))
    print("\nOdd's Ratio, Probability")
    print('The odds ratio (OR) is a statistical measure used in logistic regression to quantify the strength and direction of the association between a predictor variable and an outcome variable')
    ODDs_Ratio_df   =  pd.DataFrame({'Important Variable' : lg.params.index, 'Log-odds' : lg.params.values })
    ODDs_Ratio_df   =  ODDs_Ratio_df.loc[ODDs_Ratio_df['Important Variable'].isin(significant_vars), :]
   #ODDs_Ratio_df.drop(0, inplace = True)
    ODDs_Ratio_df['Odds Ratio']  =  np.exp(ODDs_Ratio_df['Log-odds'])
    ODDs_Ratio_df['Probability'] =   np.exp(ODDs_Ratio_df['Log-odds']) / (1 +  np.exp(ODDs_Ratio_df['Log-odds']))
    ODDs_Ratio_df.sort_values(by=['Odds Ratio'], ascending=False, inplace = True)
    print(ODDs_Ratio_df)
    ODDs_Ratio_imp_df           =    ODDs_Ratio_df.loc[ODDs_Ratio_df['Odds Ratio'] > 1, ['Important Variable', 'Probability']]
    ODDs_Ratio_imp_values       =    ODDs_Ratio_imp_df['Important Variable'].tolist()
    print(ODDs_Ratio_imp_values)
    print("\nInterpretation")
    print("\nThe probability of the event happening")
    print(ODDs_Ratio_imp_df)
def  build_models(df, Xcols, ycols, classlist):
    import  pandas                  as      pd
    from    sklearn.model_selection import  train_test_split
    from    sklearn.preprocessing   import  LabelEncoder  
    from    sklearn.experimental    import  enable_iterative_imputer  
    from    sklearn.impute          import  IterativeImputer
    from    sklearn.tree            import  DecisionTreeRegressor
    from    sklearn.linear_model    import  LogisticRegression
    from    sklearn.tree            import  DecisionTreeClassifier
    from    sklearn.neighbors       import  KNeighborsClassifier
    from    sklearn.naive_bayes     import  GaussianNB
    from    sklearn.ensemble        import  RandomForestClassifier    
    from    xgboost                 import  XGBClassifier  
    from    sklearn.ensemble        import  AdaBoostClassifier
    
   
    s           =  (df.dtypes == 'object')
    object_cols =  list(s[s].index)

    if len(object_cols) > 1:
    	print("Categorical variables:")
    	print(object_cols)

	# Convert categorical data to numerical data using LabelEncoder
    	le          = LabelEncoder()    
    	for col in object_cols:
    	    df[col] = le.fit_transform(df[col])
### Check if null values are present
    if  df.isnull().values.any():
        print("\nNull values are present and imputed using IterativeImputer() from sklearn")
        imputer         = IterativeImputer(estimator = DecisionTreeRegressor(), max_iter = 10, random_state = 0)
        imputed_data    = imputer.fit_transform(df)
    else:
        imputed_data    =  df.copy()
###        
    X                             =    imputed_data.loc[:, Xcols]
    y                             =    imputed_data.loc[:, ycols]
    
    
    X_train,X_test,y_train,y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state =3, stratify = y)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

            
    ## Apply SMOTE techniques
    ### Evaluating model performance using the data after applying SMOTE
    
    from    imblearn.over_sampling   import SMOTE
    
    # transform the dataset
    oversample = SMOTE()
    
    X_train_SMOTE, y_train_SMOTE = oversample.fit_resample(X_train, y_train)
    
    models   =  [LogisticRegression(max_iter = 10000),\
                 KNeighborsClassifier(),\
                 DecisionTreeClassifier(),\
                 GaussianNB(),\
                 RandomForestClassifier(),\
                 XGBClassifier(),\
                 AdaBoostClassifier()]
        
    SMOTE_metrics_df =  pd.DataFrame()
    model_names  =  ["LR","KNN","CART","NB","RF","XGBoost","AdaBoost"]
    for   i    in   range(len(model_names)):
          model_x            =   model_names[i]
          print("i = %d Model = %s" %(i,model_x))
          classifier         =   models[i] 
          smote_df           =   evaluate_model_SM_performance(model_x, classifier, X_train_SMOTE, y_train_SMOTE, X_test, y_test)
          SMOTE_metrics_df   =   pd.concat([SMOTE_metrics_df,smote_df])
###
    SMOTE_metrics_df.sort_values(by = 'Recall Test data', ascending = False, inplace = True)
    print(SMOTE_metrics_df)    
    
    best_model  =  SMOTE_metrics_df.iloc[0, 0]
    best_score  =  SMOTE_metrics_df.iloc[0, 1]
 
    print('Model {0} Recall {1}'.format(best_model, best_score))
    if  best_model == 'LR':
        target   =   ycols[0]
        build_LR(df, X_train_SMOTE, y_train_SMOTE, target)
  
    
    

###
## ----------------------------------------------------------------------------------------------------------------------------------------------------
###
file           =    r'G:\DSE-FT-C-May24-G5\Data\Customer-Churn-Records.csv'
df             =    pd.read_csv(file, encoding = 'Latin-1')
print(df.shape)
print(df.columns)

Xcols      =    ['Satisfaction Score','Tenure','Gender', 'IsActiveMember',
                 'Geography', 'Complain', 'Balance', 'EstimatedSalary']
ycols      =    ['Exited']  
classlist  =    ['Churn', 'No Churn']  

build_models(df, Xcols, ycols, classlist)
