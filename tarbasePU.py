import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, OneClassSVM
from baggingPU import BaggingClassifierPU
from imblearn.under_sampling import RandomUnderSampler
from puwrapper import PUWrapper
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve, auc, average_precision_score, matthews_corrcoef
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, scale
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from inspect import signature
from datetime import timedelta
from time import time
from puAdapter import PUAdapter
import pickle
from sklearn import manifold
from scipy import interp
import json 
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

# Author: Henrique Dambros <hvdambros@gmail.com>



def _load_tarbase_dataset():
    #read text file
    X = pd.read_csv(filepath_or_buffer = 'datasets/final_dataset_tarbase.att', sep = '\t')
    #put the column 'class' in a separate variable and delete the column from X
    y = X['Class']
    X = X.drop(columns = 'Class')
    # replace strings 'pos', 'neg' and 'unl' for 1, -1 and 0
    y.replace({'pos': 1, 'neg': -1, 'unl': 0},inplace=True)
    if(top_12):
         X = X.drop(columns = ['AlignLen','totalGU','totalGaps','totalMismatches','seedGaps','seedMismatches','Pos_1','Pos_3','Pos_5','Pos_8','Pos_9','Pos_10','Pos_11','Pos_12','Pos_13','Pos_14','Pos_15','Pos_16','Pos_17','Pos_18','Pos_19','Pos_20'])
    return X, y

def _plot_precision_recall_curve(precision, recall, average_precision, ylim_min, std_precision=None, std_avg_precision=None, fileName = None, colorIndex = 0):
    plt.figure('Precision-Recall curve')

    lines = []
    labels = []
    if(ylim_min == -0.05):
        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            yf = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[yf >= 0], yf[yf >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, yf[45] + 0.02))
        lines.append(l)
        labels.append('iso-f1 curves')

    pos_perc = float(len(X[y==1]))/len(X[y!=0])
    xd = [0,1]
    yd = [pos_perc,pos_perc]

    l, = plt.plot(xd, yd, color='gray', linestyle='--')

    for key in precision:
        l, = plt.plot(recall[key], precision[key], lw=2, color=colors[colorIndex])
        lines.append(l)
        if(std_avg_precision != None):
                    labels.append('{0} (auc = {1:0.2f} $\pm$ {2:0.2f})'
                      ''.format(key_translation[key.split('_')[0]], average_precision[key], std_avg_precision[key]))
        else:
            labels.append('{0} (auc = {1:0.2f})'
                      ''.format(key_translation[key.split('_')[0]], average_precision[key]))
        if(std_precision!=None):
            precision_upper = np.minimum(precision[key] + std_precision[key], 1)
            precision_lower = np.maximum(precision[key] - std_precision[key], 0)
            plt.fill_between(recall[key], precision_lower, precision_upper, color=colors[colorIndex], alpha=.2)
        colorIndex+=1
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.axis('scaled')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([ylim_min, 1.02])
    plt.xlim([-0.02, 1.02])
    title = 'Precision-Recall curve'
    if(ylim_min != -0.05):
        title = title + ' 2'
    plt.title(title)
    fig.tight_layout()
    fig.tight_layout()
    plt.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), prop=dict(size=14), fontsize='small', ncol=2)
    plt.subplots_adjust()
    if(save_plots):
        if(fileName!=None):
            title += '_'+fileName
        plt.savefig('executions/tarbase/%s/%s/%s.png' % (dr_name,folder_name, title), bbox_inches="tight")
    if(show_plots):
        plt.show()
    else:
        plt.close(fig)

def _plot_roc(tpr, fpr, roc_auc, std_tpr=None, std_auc=None, fileName = None, colorIndex = 0):
    fig = plt.figure('Receiver operating characteristic')
    plt.axis('scaled')
    for key in fpr:
        if(std_auc!=None):
            plt.plot(fpr[key], tpr[key], label='%s (auc = %0.2f $\pm$ %0.2f)' % (key_translation[key.split('_')[0]], roc_auc[key], std_auc[key]), color=colors[colorIndex])
        else:
            plt.plot(fpr[key], tpr[key], label='%s (auc = %0.2f)' % (key_translation[key.split('_')[0]], roc_auc[key]), color=colors[colorIndex])
        if(std_tpr!=None):
            tprs_upper = np.minimum(tpr[key] + std_tpr[key], 1)
            tprs_lower = np.maximum(tpr[key] - std_tpr[key], 0)
            plt.fill_between(mean_fpr[key], tprs_lower, tprs_upper, color=colors[colorIndex], alpha=.2)
        colorIndex+=1
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = 'Receiver operating characteristic'
    plt.title(title)
    plt.legend(loc="lower right")
    fig.tight_layout()
    if(save_plots):
        if(fileName!=None):
            title += '_'+fileName
        plt.savefig('executions/tarbase/%s/%s/%s.png' % (dr_name,folder_name, title), bbox_inches="tight")
    if(show_plots):
        plt.show()
    else:
        plt.close(fig)

def _plot_pvalue_matrix(p_values, methods):
    plt.rcParams['figure.figsize'] = 16,16
    fig, ax = plt.subplots()

    im = ax.imshow(p_values.to_numpy(), cmap='YlOrRd_r')
    cbar = ax.figure.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('p-vaue for t-test', rotation=-90, va="bottom")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(methods)))
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.set_ylim(len(methods)-0.5, -0.5)
    #ax.set_xlim(len(methods)-0.5, -0.5)
    ax.set_xticklabels(methods)
    ax.set_yticklabels(methods)
    ax.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    textcolors=["white", "black"]
    for i in range(len(methods)):
        for j in range(len(methods)):
            if(i==j):
                text = ""
            elif(j>i):
                text = ""
            elif(p_values.iloc[j,i]>0.001):
                text = "{:.3f}".format(p_values.iloc[i,j])
            else:
                text = "{:.2E}".format(p_values.iloc[i,j])

            ax.text(j, i, text, ha="center", va="center", size=12, color=textcolors[int(im.norm(p_values.iloc[i,j]) > 0.5)])
    if(save_plots):
        title = 'pvalue_matrix'
        plt.savefig('executions/tarbase/%s/%s/%s.png' % (dr_name,folder_name, title), bbox_inches="tight")
    if(show_plots):
        plt.show()
    else:
        plt.close(fig)

    plt.rcParams['figure.figsize'] = fig_size
    


def _dimensionality_reduction():
    # Reduce dimension to 2
    n_neighbors = 10
    dr = None
    if(load_trained_dr):
        with open('saved_objects/%s' % (dr_name), 'rb') as nca_file:
            dr = pickle.load(nca_file)
    else:
        if('nca' in dr_name):
            dr = make_pipeline(StandardScaler(),NeighborhoodComponentsAnalysis(n_components=2,random_state=None))
        elif('pca' in dr_name):
            dr = make_pipeline(StandardScaler(), PCA(n_components=2,random_state=None))
        elif('isomap' in dr_name):
            dr = make_pipeline(StandardScaler(), manifold.Isomap(n_neighbors, n_components=2))
        elif('tsne' in dr_name):
            dr = make_pipeline(StandardScaler(),manifold.TSNE(n_components=2, perplexity=50,n_iter=5000,learning_rate=300.0))
        if(dr_name not in dr_fittransform):
            dr.fit(X, y)
        if save_trained_dr:
            with open('saved_objects/dr', 'wb') as nca_file:
                pickle.dump(dr, nca_file)

    return dr

def _2d_visualization_high_dim_data(X,y,dataset_name):
    if(dr_name in dr_fittransform):
        X_embedded = dr.fit_transform(X)
    else:
        X_embedded = dr.transform(X)
    fig = plt.figure()
    ax = plt.subplot()
    if(len(X_embedded[y==0])>0):
        ax.scatter(X_embedded[y==0][:,0], X_embedded[y==0][:,1], alpha=0.1, c='grey', s=30, label='unlabeled')
    if(len(X_embedded[y==1])>0):
        ax.scatter(X_embedded[y==1][:,0], X_embedded[y==1][:,1], alpha=0.2, c='grey', s=30, label='positive')
    if(len(X_embedded[y==-1])>0):
        ax.scatter(X_embedded[y==-1][:,0], X_embedded[y==-1][:,1], alpha=0.2, c='red', s=30, label='negative')
    ax.legend(loc="lower right")
    if(ylim!=None):
        ax.set_ylim(ylim)
    if(xlim!=None):
        ax.set_xlim(xlim)
    fig.tight_layout()
    title = 'tarbase_2d_%s_%s' % (dr_name, dataset_name)
    if(save_plots):
        plt.savefig('executions/tarbase/%s/%s/%s.png' % (dr_name,folder_name, title), bbox_inches="tight")
    if(show_plots):
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
    else:
        plt.close(fig)

def _2d_visualization_probabilities(X,y,method,dr,prob=None,generate_dif_img=True):
    if(dr is None):
        with open('saved_objects/%s' % (dr_name), 'rb') as nca_file:
            dr = pickle.load(nca_file)

    if(dr_name in dr_fittransform):
        X_embedded = dr.fit_transform(X)
    else:
        X_embedded = dr.transform(X)
    if(prob is None):
        prob = results['%s_prob' % (method)]
    plt.rcParams['figure.figsize'] = fig_size
    fig = plt.figure()
    ax = plt.subplot()
    sc = ax.scatter(X_embedded[:,0], X_embedded[:,1], alpha=0.5, c=prob, s=30, cmap = 'jet_r', vmin=0, vmax=1)
    if(ylim!=None):
        ax.set_ylim(ylim)
    if(xlim!=None):
        ax.set_xlim(xlim)
    fig.tight_layout()
    cbaxes = inset_axes(ax, width="2%", height="95%", loc='center left') 
    plt.colorbar(sc, cax=cbaxes, label='probability scores given to test points', orientation='vertical')
    if(use_cross_validation):
        split = method.split('_')
        method = split[0]
        fold = split[1]
        title = 'tarbase_2d_%s_prob_score_%s_fold_%s' % (dr_name, key_translation[method], fold)
    else:
        title = 'tarbase_2d_%s_prob_score_%s' % (dr_name, key_translation[method])
    if(save_plots):
        if not os.path.exists('executions/tarbase/%s/%s/probabilities'% (dr_name, folder_name)):
            os.makedirs('executions/tarbase/%s/%s/probabilities'% (dr_name, folder_name))
        plt.savefig('executions/tarbase/%s/%s/probabilities/%s.png' % (dr_name,folder_name, title), bbox_inches="tight")
    if(show_plots):
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
    else:
        plt.close(fig)

    if generate_dif_img:
        y.replace({-1 : 0},inplace=True) #replaces -1 (negatives) for 0, as this is the probability value for negatives, positives already have value 1
        dif = y - prob
        fig = plt.figure()
        ax = plt.subplot()
        sc = ax.scatter(X_embedded[:,0], X_embedded[:,1], alpha=0.5, c=dif, s=30, cmap = 'RdBu', vmin=-1, vmax=1)
        if(ylim!=None):
            ax.set_ylim(ylim)
        if(xlim!=None):
            ax.set_xlim(xlim)
        fig.tight_layout()
        cbaxes = inset_axes(ax, width="2%", height="95%", loc='center left') 
        plt.colorbar(sc, cax=cbaxes, label='Ground truth minus probability prediction', orientation='vertical')
        title = 'tarbase_2d_%s_prob_dif_%s' % (dr_name, key_translation[method])
        if(save_plots):    
            plt.savefig('executions/tarbase/%s/%s/%s.png' % (dr_name,folder_name, title), bbox_inches="tight")
        if(show_plots):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()
        else:
            plt.close(fig)

def _random_forest_unbalanced(X_train, y_train, X_test, result_name='rfu'):
    rf = RandomForestClassifier(n_estimators = 1000,  # Use 1000 trees
        n_jobs = -1           # Use all CPU cores
    )
    rf.fit(X_train, y_train)
    # Store the scores assigned by this approach
    results[result_name+'_pred'] = rf.predict(X_test)   # The random forest's classifications
    results[result_name+'_prob'] = rf.predict_proba(X_test)[:,1]   # The random forest's probability scores

def _random_forest_balanced(X_train, y_train, X_test, result_name='rfb'):
    rus = RandomUnderSampler()
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    
    rf = RandomForestClassifier(n_estimators = 1000,  # Use 1000 trees
        n_jobs = -1           # Use all CPU cores
    )
    rf.fit(X_resampled, y_resampled)
    # Store the scores assigned by this approach
    results[result_name+'_pred'] = rf.predict(X_test)   # The random forest's classifications
    results[result_name+'_prob'] = rf.predict_proba(X_test)[:,1]   # The random forest's probability scores

def _SVC_balanced(X_train, y_train, X_test, result_name='svb'):
    rus = RandomUnderSampler()
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    svc = Pipeline([('Scaler', StandardScaler()), 
        ('classifier', SVC(gamma='scale',probability = True)) 
    ])

    #svc = SVC(gamma='scale', probability = True)
    svc.fit(X_resampled, y_resampled)
    
    results[result_name+'_pred'] = svc.predict(X_test)
    results[result_name+'_prob'] = svc.predict_proba(X_test)[:,1]

def _SVC_unbalanced(X_train, y_train, X_test, result_name='svu'):
    svc = Pipeline([('Scaler', StandardScaler()), 
        ('classifier', SVC(gamma='scale',probability = True))
    ])

    #svc = SVC(gamma='scale', probability = True)
    svc.fit(X_train, y_train)
    
    results[result_name+'_pred'] = svc.predict(X_test)
    results[result_name+'_prob'] = svc.predict_proba(X_test)[:,1]

def _bagging_PU_SVC(X_train, y_train, X_test, result_name='bsv'):
    classifier = Pipeline([('Scaler', StandardScaler()), 
        ('classifier', SVC(gamma='scale',probability = True)) 
    ])
    n_estimators = 10
    _bagging_PU(X_train, y_train, X_test,classifier,n_estimators,result_name)

def _bagging_PU_decision_trees(X_train, y_train, X_test, result_name='bdt'):
    classifier = DecisionTreeClassifier()
    n_estimators = 1000
    _bagging_PU(X_train, y_train, X_test,classifier,n_estimators, result_name)

def _bagging_PU(X_train, y_train, X_test, classifier, n_estimators, results_label):
    bc = BaggingClassifierPU(classifier,
        n_estimators = n_estimators,
        max_samples = min(len(y_train[y_train<1]),len(y_train[y_train==1])), # Balance the positives and unlabeled in each bag
        n_jobs = -1           # Use all cores
    )
    bc.fit(X_train, y_train)
    results[results_label+'_prob'] = bc.predict_proba(X_test)[:,1]
    results[results_label+'_pred'] = bc.predict(X_test)

def _elkan_noto(X_train, y_train, X_test, result_name='enm'):
    base1 = Pipeline([('transformer', RBFSampler(n_components=100, random_state=None)), ('Scaler', StandardScaler()),
        ('classifier', LogisticRegression(solver='lbfgs')) 
        #('classifier', SVC(gamma='auto', probability=True))
    ])
    #base1 = RandomForestClassifier(n_estimators = 1000,  # Use 1000 trees
    #    n_jobs = -1           # Use all CPU cores
    #)
    clf = PUAdapter(base1, hold_out_ratio=0.2)
    clf.fit(X_train.to_numpy(),y_train.to_numpy())
    results[result_name+'_prob'] = clf.predict_proba(X_test.to_numpy())
    results[result_name+'_pred'] = clf.predict(X_test.to_numpy())
    results[result_name+'_pred'] = np.where(results[result_name+'_pred']==-1, 0, results[result_name+'_pred']) #replaces -1 with 0

def _elkan_noto2(X_train, y_train, X_test, result_name='ent'):
    base1 = Pipeline([('transformer', RBFSampler(n_components=100, random_state=None)), # should specify random_state to reproduce the result
        ('classifier', LogisticRegression(solver='lbfgs')) # Any classifier can be used.  To use the method predict_proba, you need to use classifiers which implements predict_proba
        #('classifier', SVC(gamma='auto', probability=True))
    ])
    #base1 = RandomForestClassifier(n_estimators = 1000,  # Use 1000 trees
    #    n_jobs = -1           # Use all CPU cores
    #)

    clf = PUWrapper(base1).fit(X_train,y_train)
    results[result_name+'_prob'] = clf.predict_proba(X_test)[:,1]
    results[result_name+'_pred'] = clf.predict(X_test)

def _two_step(X_train, y_train, X_test, result_name = 'tst'):
    #implementation from roy wright in https://roywrightme.wordpress.com/2017/11/16/positive-unlabeled-learning/
    # Create a new target vector, with 1 for positive, -1 for unlabeled, and
    # 0 for "reliable negative" (there are no reliable negatives to start with)
    ys = 2 * y_train - 1

    rf = RandomForestClassifier(n_estimators = 1000,  # Use 1000 trees
        n_jobs = -1           # Use all CPU cores
    )
    rus = RandomUnderSampler()
    X_resampled, y_resampled = rus.fit_resample(X_train, ys)
    rf.fit(X_resampled, y_resampled)
    pred = rf.predict_proba(X_train)[:,1]   # The random forest's scores

    # Find the range of scores given to positive data points
    range_P = [min(pred * (ys > 0)), max(pred * (ys > 0))]

    # STEP 1
    # If any unlabeled point has a score above all known positives,
    # or below all known positives, label it accordingly
    iP_new = ys[(ys < 0) & (pred >= range_P[1])].index
    iN_new = ys[(ys < 0) & (pred <= range_P[0])].index
    ys.loc[iP_new] = 1
    ys.loc[iN_new] = 0

    # Classifier to be used for step 2
    rf2 = RandomForestClassifier(n_estimators = 1000, n_jobs = -1)
    #rf2 = SVC(gamma='scale',probability = True)

    # Limit iterations (this is arbitrary, but
    # otherwise this approach can take a very long time)
    for i in range(3):
        # If step 1 didn't find new labels, we're done
        if len(iP_new) + len(iN_new) == 0 and i > 0:
            break
    
        #print('Step 1 labeled %d new positives and %d new negatives.' % (len(iP_new), len(iN_new)))
        #print('Doing step 2... ', end = '')
    
        # STEP 2
        # Retrain on new labels and get new scores
        rf2.fit(X_train, ys)
        pred = rf2.predict_proba(X_train)[:,-1]
    
        # Find the range of scores given to positive data points
        range_P = [min(pred * (ys > 0)), max(pred * (ys > 0))]
    
        # Repeat step 1
        iP_new = ys[(ys < 0) & (pred >= range_P[1])].index
        iN_new = ys[(ys < 0) & (pred <= range_P[0])].index
        ys.loc[iP_new] = 1
        ys.loc[iN_new] = 0
    
    
    # Lastly, get the scores assigned by this approach
    results[result_name+'_prob'] = rf2.predict_proba(X_test)[:,-1]
    results[result_name+'_pred'] = rf2.predict(X_test)
    results[result_name+'_pred'] = np.where(results[result_name+'_pred']==-1, 0, results[result_name+'_pred']) #replaces -1 with 0

def _train_one_class_svm(X):
    ocsvm = OneClassSVM(kernel="rbf",gamma='scale')
    ocsvm.fit(scale(X))
    return ocsvm

def _one_class_svm(X_train, y_train, X_test, result_name='ocs'):
    #use _train_one_class_svm first
    #log = Pipeline([('transformer', RBFSampler(n_components=100, random_state=None)), ('Scaler', StandardScaler()),
    #                ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000))])

    log = Pipeline([('Scaler', StandardScaler()),
                    ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000))])

    log.fit(ocsvm.score_samples(scale(X_train)).reshape(-1, 1) , y_train)

    results[result_name+'_prob'] = log.predict_proba(ocsvm.score_samples(scale(X_test)).reshape(-1, 1))[:,1]
    results[result_name+'_pred'] = ocsvm.predict(scale(X_test))
    results[result_name+'_pred'] = np.where(results[result_name+'_pred']==-1, 0, results[result_name+'_pred']) #replaces -1 with 0

def _train_isolation_forest(X):
    isf = IsolationForest(n_estimators = 1000, behaviour='new', contamination='auto')
    isf.fit(X)
    return isf

def _isolation_forest(X_train, y_train, X_test, result_name='isf'):
    #use _train_isolation_forest first
    #log = Pipeline([('transformer', RBFSampler(n_components=100, random_state=None)), ('Scaler', StandardScaler()),
    #                ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000))])

    log = Pipeline([('Scaler', StandardScaler()),
                    ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000))])

    log.fit(isf.decision_function(X_train).reshape(-1, 1) , y_train)
    
    results[result_name+'_pred'] = isf.predict(X_test)
    results[result_name+'_pred'] = np.where(results[result_name+'_pred']==-1, 0, results[result_name+'_pred']) #replaces -1 with 0
    results[result_name+'_prob'] = log.predict_proba(isf.decision_function(X_test).reshape(-1, 1))[:,1]

def cv_training(X, y, training_method, training_label, remove_unlabeled=False):
    i = 1
    if(remove_unlabeled):
        X = X[y!=0]
        y = y[y!=0]

    for train, test in cv.split(X,y):
        y_test = y.iloc[test]
        y_train = y.iloc[train]
        X_train = X.iloc[train]
        if(equalize_pos_neg):
            X_train, y_train = _subsample(X_train,y_train)
        if(remove_all_negatives and not remove_unlabeled):
            X_train = X_train[y_train!=-1]
            y_train =  y_train[y_train!=-1]
            training_method(X_train, y_train, X.iloc[test][y_test!=0], training_label+"_"+str(i))
        else:
            if(remove_unlabeled):
                y_train =  y_train[y_train!=-1].append(y_train[y_train==-1].replace({-1:0})) #sets negative label as 0 instead of -1
                training_method(X_train, y_train, X.iloc[test], training_label+"_"+str(i))
            else:            
                if(use_positives_as_unlabeled):
                    y_train =  y_train[y_train!=1].append(y_train[y_train==1].replace({1:0})) #sets all positives labels (1) as unlabeled (0) for training
                    y_train =  y_train[y_train!=-1].append(y_train[y_train==-1].replace({-1:1}))
                else:
                    y_train =  y_train[y_train!=-1].append(y_train[y_train==-1].replace({-1:0})) #sets all negative labels (-1) as unlabeled (0) for training
                training_method(X_train, y_train, X.iloc[test][y_test!=0], training_label+"_"+str(i))
                if(use_positives_as_unlabeled):
                    j=0
                    for result in results[training_label+"_"+str(i)+"_prob"]:
                        results[training_label+"_"+str(i)+"_prob"][j] = 1-result
                        j+=1
                    results[training_label+"_"+str(i)+"_pred"] = np.where(results[training_label+"_"+str(i)+"_pred"]==1,-2,results[training_label+"_"+str(i)+"_pred"])
                    results[training_label+"_"+str(i)+"_pred"] = np.where(results[training_label+"_"+str(i)+"_pred"]==0,1,results[training_label+"_"+str(i)+"_pred"])
                    results[training_label+"_"+str(i)+"_pred"] = np.where(results[training_label+"_"+str(i)+"_pred"]==-2,0,results[training_label+"_"+str(i)+"_pred"])
        results[training_label+"_"+str(i)+"_testLabels"] = y_test[y_test!=0].to_numpy()
        results[training_label+"_"+str(i)+"_testAtt"] = X.iloc[test][y_test!=0].index._data
        results[training_label+"_"+str(i)+"_trainAtt"] = X_train.index._data
        results[training_label+"_"+str(i)+"_trainLabels"] = y_train.to_numpy()
        i+=1

def _subsample(X,y):
    if(len(y[y==-1])==0):
        if(len(y[y==1])>len(y[y==0])):
            c = 1
            mc = 0
        else:
            c = 0
            mc = 1
    else:
        if(len(y[y==1])>len(y[y==-1])):
            c = 1
            mc = -1
        else:
            c = -1
            mc = 1
    Xc = X[y==c]
    yc = y[y==c]
    if(float(len(y[y==mc]))/len(yc)<1):
        _, Xc, _, yc = train_test_split(Xc, yc, test_size= float(len(y[y==mc]))/len(yc))
    Xr = Xc.append(X[y!=c])
    yr = yc.append(y[y!=c])
    return Xr, yr

if __name__ == '__main__':
    fig_size = 19,9
    show_plots = False
    save_plots = True
    folder_name = 'new' #default folder name to save execution, used if create_custom_folder_name is set to False
    create_custom_folder_name = False
    n_splits_cv = 10
    shuffle_cv = False #not recommended without 'remove_duplicates'

    num_unlabeled_to_use = 0
    
    use_positives_as_unlabeled = False
    remove_negative_cluster = False
    remove_all_negatives = False
    equalize_pos_neg = False
    load_execution_from_File = False #if loading from file, remove_duplicates and top12 should have the same values as the ones used in the execution being loaded
    top_12 = False #top 12 features, described in MENDOZA, M. R. et al. RFMirTarget: predicting human microRNA target genes with a random forest classifier (2013)
    generate_probabilites_image = False
    remove_duplicates = True
    remove_methods_on_load = False

    dr_fittransform = ['tsne']
    load_trained_dr = True
    save_trained_dr = False
    
    run_multiple_executions = False #if True, create_custom_folder_name should be True as well
    if(run_multiple_executions):
        # atributes on each line [num_unlabeled_to_use,use_positives_as_unlabeled,remove_negative_cluster,remove_all_negatives,equalize_pos_neg]
        executions_parameters = [
            [0, False,False, False, False],
            #[5000, False,False, True, False],
            #[6000, False,False, True, False],
            #[7000, False,False, True, False],
            [10000, False,False, True, False],
            [0, False,True, False, False],
            #[100, False,True, False, False],
            #[1000, False,True, False, False],
            [10000, False,True, False, False],
            #[100000, False,True, False, False],
            [0, True,False, False, False],
            #[100, True,False, False, False],
            [1000, True,False, False, False],
            [10000, True,False, False, False],
            #[100000, True,False, False, False],
            [0, False,False, False, True],
            #[100, False,False, False, True],
            [1000, False,False, False, True],
            [10000, False,False, False, True],
            #[100000, False,False, False, True],
            ]
    else:
        executions_parameters = [[num_unlabeled_to_use,use_positives_as_unlabeled,remove_negative_cluster,remove_all_negatives,equalize_pos_neg]]

    if(load_execution_from_File):
        if(remove_methods_on_load):
            methods_to_use = [
                #'rfb',
                #'svb',
                'bdt',
                'bsv',
                #'enm',
                #'ent',
                'tst',
                #'rfu',
                'rfn',
                'svn',
                #'svu',
                'ocs',
                'isf',
                #'noc',
                #'nif',
                ]    

    #2d visualization method constansts. Only one should be uncommented
    #nca constansts
    #dr_name = 'nca3'
    #ylim = [-900., 600.]
    #xlim = [-1300., 1250.]

    dr_name = 'nca4'
    ylim = [-1250., 750.]
    xlim = [-1800., 1800.]
    neg_cluster_x = [800,1200]
    neg_custer_y = [-150,500]

    #dr_name = 'nca5'
    #ylim = [-8000., 9500.]
    #xlim = [-6800., 5500.]

    #dr_name = 'nca6'
    #ylim = [-2000., 4000.]
    #xlim = [-4500., 4500.]

    #dr_name = 'nca7'
    #ylim = None
    #xlim = None
    
    #pca constants
    #dr_name = 'pca'
    #ylim = [-6., 7.]
    #xlim = [-8., 9.]

    #dr_name = 'pca2'
    #ylim = [-6., 7.]
    #xlim = [-8., 9.]

    #dr_name = 'pca3'
    #ylim = None
    #xlim = None
    
    #isomap constants
    #dr_name = 'isomap'
    #ylim = [-20., 25.]
    #xlim = [-25., 40.]

    #tsne constants
    #dr_name = 'tsne'
    #ylim = None
    #xlim = None

    #colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
    colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#b15928','#ffff99'] #colors used in plots
    numColors = 12
    cmap = LinearSegmentedColormap.from_list("", colors)

    for exec_param in executions_parameters:
        num_unlabeled_to_use = exec_param[0]
        use_positives_as_unlabeled = exec_param[1]
        remove_negative_cluster = exec_param[2]
        remove_all_negatives = exec_param[3]
        equalize_pos_neg = exec_param[4]

        if(create_custom_folder_name):
            if(remove_all_negatives):
                folder_name = 'no_neg '
            elif(remove_negative_cluster):
                folder_name = 'neg_cluster '
            elif(use_positives_as_unlabeled):
                folder_name = 'pos_as_unl '
            elif(equalize_pos_neg):
                folder_name = 'pos=neg '
            else:
                folder_name = ''

            folder_name += str(num_unlabeled_to_use) + 'unl'

        dr = None #dimensionality reduction model
        print('loading dataset... ', end = '', flush=True)
        start = time()
    
        plt.rcParams['figure.figsize'] = fig_size   # graph dimensions
        plt.rcParams['font.size'] = 14         # graph font size

        key_translation = {'rfb': 'Random Forest with Subsampling',
                           'svb': 'SVM with Subsampling',
                           'bdt': 'BaggingPU Decision Trees',
                           'bsv': 'BaggingPU SVM',
                           'enm': 'Elkan Noto',
                           'ent': 'Elkan Noto 2',
                           'tst': 'Two Step',
                           'rfu': 'Random Forest',
                           'rfn': 'Baseline Random Forest',
                           'svn': 'Baseline SVM',
                           'svu': 'SVM',
                           'ocs': 'One-Class SVM',
                           'isf': 'Isolation Forest',
                           'noc': 'Novelty One-Class SVM',
                           'nif': 'Novelty Isolation Forest',
                           'tes': 'Truth'}
        key_prob_tr = dict()
        for key in key_translation:
            key_prob_tr[key+'_prob'] = key_translation[key]
        key_pred_tr = dict()
        for key in key_translation:
            key_pred_tr[key+'_pred'] = key_translation[key]


        X, y = _load_tarbase_dataset()

        if(remove_duplicates):
            #removes duplicates in each class, keeping the first
            dup_p = X[y==1].duplicated(keep='first')
            dup_u = X[y==0].duplicated(keep='first')
            dup_n = X[y==-1].duplicated(keep='first')
            Xpn = X[y==1][dup_p==False].append(X[y==-1][dup_n==False])
            ypn = y[y==1][dup_p==False].append(y[y==-1][dup_n==False])
            #removes all duplicates between positives and negatives
            dup = Xpn.duplicated(keep=False) 
            Xpn = Xpn[dup==False]
            ypn = ypn[dup==False]
            X = Xpn.append(X[y==0][dup_u==False])
            y = ypn.append(y[y==0][dup_u==False])
            #removes duplicates of positives and negatives in the unlabeled class
            dup = X.duplicated(keep='first')
            X = X[dup==False]
            y = y[dup==False]


        if(not load_execution_from_File):
            num_unlabeled = len(y[y==0])
            if(num_unlabeled_to_use>0 and num_unlabeled_to_use<num_unlabeled):
                X_train0 = X[y==0]    #unlabeled data
                y_train0 = y[y==0]
                _, X_train0, _, y_train0 = train_test_split(X_train0, y_train0, test_size= float(num_unlabeled_to_use)/len(y_train0)) #removing some unlabeled points

                X = X[y!=0].append(X_train0)
                y = y[y!=0].append(y_train0)
            elif(num_unlabeled_to_use<=0):
                X = X[y!=0]
                y = y[y!=0]

            if(shuffle_cv):
                cv = StratifiedKFold(n_splits=n_splits_cv, shuffle=True) #StratifiedKFold is a cross validation scheme that maintains percentages of each class
            else:
                cv = StratifiedKFold(n_splits=n_splits_cv, shuffle=False)
            database_txt = 'dataset: %d positives %d unlabeled %d negatives' % (len(y[y==1]),len(y[y==0]), len(y[y==-1]))
            print(database_txt)
            if(save_plots):
                if not os.path.exists('executions/tarbase/%s/%s'% (dr_name, folder_name)):
                    os.makedirs('executions/tarbase/%s/%s'% (dr_name, folder_name))

                with open('executions/tarbase/%s/%s/output.txt' % (dr_name, folder_name), 'w') as f1_file:
                    f1_file.write(database_txt+'\n')
                    f1_file.write('use_positives_as_unlabeled:'+ str(use_positives_as_unlabeled) +'\n')
                    f1_file.write('remove_negative_cluster:'+ str(remove_negative_cluster) +'\n')
                    f1_file.write('remove_all_negatives:'+ str(remove_all_negatives) +'\n')
                    f1_file.write('equalize_pos_neg:'+ str(equalize_pos_neg) +'\n')
                    f1_file.write('remove_duplicates:'+ str(remove_duplicates) +'\n')

    
            results = dict()
            elapsed = time() - start
            print('[done] execution time: %s' % str(timedelta(seconds=elapsed)))
       
            print('generating 2d visualization... ', end='', flush=True)
            start = time()
            dr = _dimensionality_reduction()
            if(remove_negative_cluster):
                delete_X = []
                delete_y = []
                X_embedded = dr.transform(X)
                for i in range(len(X_embedded)):
                    if(X_embedded[i][0]>neg_cluster_x[0] and X_embedded[i][0]<neg_cluster_x[1]):
                        if(X_embedded[i][1]>neg_custer_y[0] and X_embedded[i][1]<neg_custer_y[1]):
                            delete_X.append(X.index[i])
                            delete_y.append(y.index[i])

                X.drop(delete_X, inplace=True)
                y.drop(delete_y, inplace=True)

            _2d_visualization_high_dim_data(X,y, dataset_name='dataset_all')
            if(num_unlabeled_to_use>0 and dr_name not in dr_fittransform):
                _2d_visualization_high_dim_data(X[y==0],y[y==0], dataset_name='dataset_unlabeled')

            elapsed = time() - start
            print('[done] execution time: %s' % str(timedelta(seconds=elapsed)))


            print('training baseline random forest ... ', end='', flush=True)
            start = time()
            cv_training(X,y,_random_forest_unbalanced,'rfn',remove_unlabeled=True)
            elapsed = time() - start
            print('[done] execution time: %s' % str(timedelta(seconds=elapsed)))

            #print('training random forest... ', end='', flush=True)
            #start = time()
            #cv_training(X,y,_random_forest_unbalanced,'rfu')
            #elapsed = time() - start
            #print('[done] execution time: %s' % str(timedelta(seconds=elapsed)))
    
            #print('training random forest with subsampling... ', end='', flush=True)
            #start = time()
            #cv_training(X,y,_random_forest_balanced,'rfb')
            #elapsed = time() - start
            #print('[done] execution time: %s' % str(timedelta(seconds=elapsed)))

            #print('training baseline SVM... ', end='', flush=True)
            #start = time()
            #cv_training(X,y,_SVC_unbalanced,'svn',remove_unlabeled=True)
            #elapsed = time() - start
            #print('[done] execution time: %s' % str(timedelta(seconds=elapsed)))
    
            #print('training SVM with subsampling... ', end='', flush=True)
            #start = time()
            #cv_training(X,y,_SVC_balanced,'svb')
            #elapsed = time() - start
            #print('[done] execution time: %s' % str(timedelta(seconds=elapsed)))

            #print('training baggingPU decision trees... ', end='', flush=True)
            #start = time()
            #cv_training(X,y,_bagging_PU_decision_trees,'bdt')
            #elapsed = time() - start
            #print('[done] execution time: %s' % str(timedelta(seconds=elapsed)))
    
            #print('training baggingPU SVM... ', end='', flush=True)
            #start = time()
            #cv_training(X,y,_bagging_PU_SVC,'bsv')
            #elapsed = time() - start
            #print('[done] execution time: %s' % str(timedelta(seconds=elapsed)))

            #print('training two step... ', end='', flush=True)
            #start = time()
            #cv_training(X,y,_two_step,'tst')
            #elapsed = time() - start
            #print('[done] execution time: %s' % str(timedelta(seconds=elapsed)))

            #print('training one class svm... ', end='', flush=True)
            #start = time()
            #if(remove_all_negatives):
            #    Xoc = X[y!=-1]
            #    ocsvm = _train_one_class_svm(Xoc)
            #    cv_training(X,y,_one_class_svm,'ocs')
            #else:
            #    ocsvm = _train_one_class_svm(X)
            #    cv_training(X,y,_one_class_svm,'ocs',remove_unlabeled=True)
            #elapsed = time() - start
            #print('[done] execution time: %s' % str(timedelta(seconds=elapsed)))

            #print('training isolation forest... ', end='', flush=True)
            #start = time()
            #if(remove_all_negatives):
            #    Xoc = X[y!=-1]
            #    isf = _train_isolation_forest(Xoc)
            #    cv_training(X,y,_isolation_forest,'isf')
            #else:
            #    isf = _train_isolation_forest(X)
            #    cv_training(X,y,_isolation_forest,'isf',remove_unlabeled=True)
            #elapsed = time() - start
            #print('[done] execution time: %s' % str(timedelta(seconds=elapsed)))

            #print('training novelty one class svm... ', end='', flush=True)
            #start = time()
            #ocsvm = _train_one_class_svm(X[y==1])
            #cv_training(X,y,_one_class_svm,'noc',remove_unlabeled=True)
            #elapsed = time() - start
            #print('[done] execution time: %s' % str(timedelta(seconds=elapsed)))

            #print('training novelty isolation forest... ', end='', flush=True)
            #start = time()
            #isf = _train_isolation_forest(X[y==1])
            #cv_training(X,y,_isolation_forest,'nif',remove_unlabeled=True)
            #elapsed = time() - start
            #print('[done] execution time: %s' % str(timedelta(seconds=elapsed)))

            #print('training elkan noto... ', end='', flush=True)
            #start = time()
            #cv_training(X,y,_elkan_noto,'enm')
            #elapsed = time() - start
            #print('[done] execution time: %s' % str(timedelta(seconds=elapsed)))

            #print('training elkan noto 2... ', end='', flush=True)
            #start = time()
            #cv_training(X,y,_elkan_noto2,'ent')
            #elapsed = time() - start
            #print('[done] execution time: %s' % str(timedelta(seconds=elapsed)))

            #print('training SVM... ', end='', flush=True) #takes too long with more than 10000 unlabeled
            #start = time()
            #cv_training(X,y,_SVC_unbalanced,'svu') 
            #elapsed = time() - start
            #print('[done] execution time: %s' % str(timedelta(seconds=elapsed)))


        open('executions/tarbase/%s/%s/scores.txt' % (dr_name, folder_name), 'w').close()

        if(load_execution_from_File):
            with open('executions/tarbase/%s/%s/execution.txt' % (dr_name, folder_name), 'r') as env_file:
                results = json.load(env_file)
                for k in results.keys():
                    if(isinstance(results[k], list)):
                        results[k] = np.array(results[k])

            if(remove_methods_on_load):
                for k in list(results.keys()):
                    if(not any(x in k for x in methods_to_use)):
                        results.pop(k)

        if(generate_probabilites_image):
            for k in results.keys():
                if('_prob' in k):
                    prob = results[k]
                    fold = k.split('_')[1]
                    method = k.replace('_prob','')
                    key_testatt = k.replace('prob','testAtt')
                    Xt = X.iloc[results[key_testatt]]
                    yt = y.iloc[results[key_testatt]]
                    _2d_visualization_probabilities(Xt,yt,('tes_%s'% fold),dr,yt,False)
                    _2d_visualization_probabilities(Xt,yt,method,dr,prob,False)
                        

        precision = dict()
        recall = dict()
        average_precision = dict()
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        tprs = dict()
        roc_aucs = dict()
        mean_fpr = dict()
        precisions = dict()
        pr_auc = dict()
        pr_aucs = dict()
        mean_recall = dict()
        f1s = dict()
        f1ns = dict()
        mccs = dict()
        for key in results.keys():
            if('testLabels' in key):
                #results[key].replace({-1 : 0},inplace=True) 
                results[key] = np.where(results[key]==-1, 0, results[key]) #replaces negative labels (-1) with the value 0 expected by the scikit learn functions
                method = key.split('_')[0]
                tprs[method] = []
                roc_aucs[method] = []
                mean_fpr[method] = np.linspace(0, 1, 100)
                precisions[method] = []
                pr_aucs[method] = []
                mean_recall[method] = np.linspace(0, 1, 100)
                f1s[method] = []
                f1ns[method] = []
                mccs[method] = []


        for key in results.keys():
            if('pred' in key):
                f1s[key.split('_')[0]].append(f1_score(results[key.replace('pred','testLabels')],results[key]))
                resuts_test = results[key.replace('pred','testLabels')]
                resuts_test = np.where(resuts_test==0, 2,resuts_test)
                resuts_test = np.where(resuts_test==1, 0,resuts_test)
                resuts_test = np.where(resuts_test==2, 1,resuts_test)
                resultsLab = results[key]
                resultsLab = np.where(resultsLab==0, 2,resultsLab)
                resultsLab = np.where(resultsLab==1, 0,resultsLab)
                resultsLab = np.where(resultsLab==2, 1,resultsLab)
                f1ns[key.split('_')[0]].append(f1_score(resuts_test, resultsLab))
                mccs[key.split('_')[0]].append(matthews_corrcoef(results[key.replace('pred','testLabels')],results[key]))
            if("prob" in key):
                precision[key], recall[key], _ = precision_recall_curve(results[key.replace('prob','testLabels')], results[key])
                average_precision[key] = average_precision_score(results[key.replace('prob','testLabels')], results[key])
                pr_auc[key] = auc(recall[key], precision[key])
                
                reversed_recall = np.fliplr([recall[key]])[0]
                reversed_precision = np.fliplr([precision[key]])[0]
                precisions[key.split('_')[0]].append(interp(mean_recall[key.split('_')[0]], reversed_recall, reversed_precision))
                pr_aucs[key.split('_')[0]].append(pr_auc[key])

                fpr[key], tpr[key], _ = roc_curve(results[key.replace('prob','testLabels')], results[key])
                roc_auc[key] = auc(fpr[key], tpr[key])

                tprs[key.split('_')[0]].append(interp(mean_fpr[key.split('_')[0]], fpr[key], tpr[key]))
                tprs[key.split('_')[0]][-1][0] = 0.0
                roc_aucs[key.split('_')[0]].append(roc_auc[key])

        mean_f1 = dict()
        for key in f1s.keys():
            mean_f1[key] = sum(f1s[key])/len(f1s[key])
            f1 = '%s f1 score - classe positiva: %f' % (key_translation[key.split('_')[0]], mean_f1[key])
            print (f1)
            if(save_plots):
                with open('executions/tarbase/%s/%s/scores.txt' % (dr_name, folder_name), 'a') as f1_file:
                    f1_file.write(f1+'\n')

        mean_f1n = dict()
        for key in f1ns.keys():
            mean_f1n[key] = sum(f1ns[key])/len(f1ns[key])
            f1n = '%s f1 score - classe negativa: %f' % (key_translation[key.split('_')[0]], mean_f1n[key])
            print (f1n)
            if(save_plots):
                with open('executions/tarbase/%s/%s/scores.txt' % (dr_name, folder_name), 'a') as f1n_file:
                    f1n_file.write(f1n+'\n')

        mean_mcc = dict()
        for key in mccs.keys():
            mean_mcc[key] = sum(mccs[key])/len(mccs[key])
            mcc = '%s mcc score: %f' % (key_translation[key.split('_')[0]], mean_mcc[key])
            print (mcc)
            if(save_plots):
                with open('executions/tarbase/%s/%s/scores.txt' % (dr_name, folder_name), 'a') as mcc_file:
                    mcc_file.write(mcc+'\n')

        mean_tpr = dict()
        mean_roc_auc = dict()
        std_roc_auc = dict()
        std_tpr = dict()
        for key in mean_fpr.keys():
            mean_tpr[key] = np.mean(tprs[key], axis=0)
            mean_tpr[key][-1] = 1.0
            mean_roc_auc[key] = auc(mean_fpr[key], mean_tpr[key])
            std_roc_auc[key] = np.std(roc_aucs[key])
            std_tpr[key] = np.std(tprs[key], axis=0)
            fpr[key+'_avg'] = mean_fpr[key]
            tpr[key+'_avg'] = mean_tpr[key]
            roc_auc[key+'_avg'] = mean_roc_auc[key]

        mean_precision = dict()
        mean_pr_auc = dict()
        std_precision = dict()
        std_pr_auc = dict()
        for key in mean_recall.keys():
            mean_precision[key] = np.mean(precisions[key], axis=0)
            mean_pr_auc[key] = auc(mean_recall[key], mean_precision[key])
            std_pr_auc[key] = np.std(pr_aucs[key])
            std_precision[key] = np.std(precisions[key], axis=0)
            recall[key+'_avg'] = mean_recall[key]
            precision[key+'_avg'] = mean_precision[key]
            average_precision[key+'_avg'] = mean_pr_auc[key]

        p_values = pd.DataFrame(np.zeros(shape=(len(roc_aucs),len(roc_aucs))), columns=roc_aucs.keys(), index=roc_aucs.keys())
        for k in roc_aucs.keys():
            for l in roc_aucs.keys():
                if(k==l):
                    p_values[k][l] = 1
                else:
                    _, p_values[k][l] = stats.ttest_ind(roc_aucs[k],roc_aucs[l])

        
        _plot_pvalue_matrix(p_values,[key_translation[x] for x in roc_aucs.keys()])

        y_cut = 0.7
        _plot_precision_recall_curve(mean_precision, mean_recall, mean_pr_auc, -0.05, None, std_pr_auc)
        _plot_precision_recall_curve(mean_precision, mean_recall, mean_pr_auc, y_cut, None, std_pr_auc)
        i=0
        for k in mean_precision:
            _plot_precision_recall_curve({k: mean_precision[k]},{k: mean_recall[k]}, {k: mean_pr_auc[k]}, y_cut, {k: std_precision[k]}, {k: std_pr_auc[k]}, k, i)
            i+=1
            
        _plot_roc(mean_tpr, mean_fpr, mean_roc_auc, None, std_roc_auc)
        i=0
        for k in mean_tpr:
            _plot_roc({k: mean_tpr[k]},{k: mean_fpr[k]}, {k: mean_roc_auc[k]}, {k: std_tpr[k]}, {k: std_roc_auc[k]}, k, i)
            i+=1

        if(not load_execution_from_File):
            for key in results.keys():
                if isinstance(results[key], np.ndarray):
                    results[key] = results[key].tolist()
            with open('executions/tarbase/%s/%s/execution.txt' % (dr_name, folder_name), 'w') as env_file:
                env_file.write(json.dumps(results))





