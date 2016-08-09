'''
Contains useful machine learning functions.
'''

def optimisePars(mva, points, data , classes, fraction=0.7, score = 'log_loss', cvs=5):
    '''
    Funtion to optimise hyper-parameters. Follows sklearn example:
    "example-model-selection-grid-search-digits-py"
    
    Arguments:
    mva - multivariate method to optimise
    points - dictionary that holds optimisation points (hyper-parameters)
    data - your training data
    classes - true categories/classes/labels 
    fraction - fraction of training/test split
    score - score function to optimies the classifier
    cvs - number of cross-validation folds or cross-validation generator
    
    Returns:
    clf - GridSearchCV classifier.

    To-Do:
    - classification report does not exactly work every time. 
    '''
    import time
    print("# Tuning hyper-parameters for log_loss score")
    
    # Splits data
    data_train, data_test, classes_train, classes_test =  train_test_split(
    data, classes, test_size=fraction, random_state=0)
    s =  time.time()
    clf = GridSearchCV(mva, points, cv=cvs,
                       scoring=score, n_jobs=4, 
                       verbose=2)
                       
    clf.fit(data_train, classes_train)

    print('GridSearch completed after ', (time.time()-s)/60.0, ' minutes.')
    print()
    print("Best parameters set found on training set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on training set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full training set.")
    print("The scores are computed on the full test set.")
    print()
    y_true, y_pred = classes_test, clf.predict_proba(data_test)
    print("Log loss score on test sample: ", log_loss(y_true, y_pred))
    
    return clf