#Function to plot histogram with errorbars
def _hist_errorbars( data, xerrs=True, *args, **kwargs) :
  """
  Plot a histogram with error bars. 
  Accepts any kwarg accepted by either numpy.histogram or pyplot.errorbar
  """
  import numpy as np
  import matplotlib.pyplot as plt
  import inspect

  # pop off normed kwarg, since we want to handle it specially
  norm = False
  if 'normed' in list(kwargs.keys()) :
    norm = kwargs.pop('normed')

  # retrieve the kwargs for numpy.histogram
  histkwargs = {}
  for key, value in kwargs.items() :
    if key in inspect.getargspec(np.histogram).args :
      histkwargs[key] = value

  histvals, binedges = np.histogram( data, **histkwargs )
  yerrs = np.sqrt(histvals)

  if norm :
    nevents = float(sum(histvals))
    binwidth = (binedges[1]-binedges[0])
    histvals = histvals/nevents/binwidth
    yerrs = yerrs/nevents/binwidth

  bincenters = (binedges[1:]+binedges[:-1])/2

  if xerrs :
    xerrs = (binedges[1]-binedges[0])/2
  else :
    xerrs = None

  # retrieve the kwargs for errorbar
  ebkwargs = {}
  for key, value in kwargs.items() :
    if key in inspect.getargspec(plt.errorbar).args :
      ebkwargs[key] = value
    if key == 'color':
      ebkwargs['ecolor'] = value
      ebkwargs['mfc'] = value
    if key == 'label':
      ebkwargs['label'] = value

  out = plt.errorbar(bincenters, histvals, yerrs, xerrs, fmt="o", mec='black', ms=8, **ebkwargs)


  if 'log' in list(kwargs.keys()) :
    if kwargs['log'] :
      plt.yscale('log')

  if 'range' in list(kwargs.keys()) :
    plt.xlim(*kwargs['range'])

  return out

#Function to plot classifier output
def plot_classifier_output( pred_train, pred_test, y_train, y_test,
              multipagepdf=None, bins = None, normalised = True ):
  """
  Plots classifier output. 
  Works nicely with yandex/rep folding classifiers
  
  Arguments:
  pred_train: predictions for the trainings sample from the classifier.

  pred_test: predictions for the test sample from the classifier.

  y_train: true labels for the train sample. 

  y_test: true labels for the test sameple.

  multipagepdf: argument to pass multipagepdf instance.

  bins: binning for the resulting plot.

  normalised: control whether the plots are drawn normalised.
  """
  import numpy as np
  import matplotlib.pyplot as plt
  import inspect
  #Set binning
  binning = ( bins if bins is not None
              else np.linspace(np.min(pred_train), np.max(pred_train), 51) )

  #Set label to signal (=1) if label is None (for application )
  if y_train is None:
      y_train = np.ones(len(pred_train))
  if y_test is None:
      y_test = np.ones(len(pred_test))

  #Plot training sample
  plt.hist(pred_train[y_train<0.5], bins = binning, normed=normalised, histtype='stepfilled', color='b', alpha = 0.5,
           linewidth=0, label="Training Background")
  plt.hist(pred_train[y_train>0.5], bins = binning, normed=normalised, color='r', alpha = 0.5,
           linewidth=0, label="Training Signal", histtype='stepfilled')

  #Plot test sample
  _hist_errorbars( pred_test[y_test<0.5], xerrs=True, bins = binning, normed=normalised,
                  color='b', label="Test Background")
  _hist_errorbars( pred_test[y_test>0.5], xerrs=True, bins = binning, normed=normalised,
                  color='r', label="Test Signal")


  #plt.title("Classifier Output - Signal vs. Background and Test vs Training", fontsize=23)

  plt.xlabel("Classifier Output", fontsize=23)
  plt.ylabel("Normalised Events" if normalised else "Events", fontsize=23)

  #Restrict plot to binning area
  plt.xlim(binning[0], binning[-1])

  #Plot legend
  plt.legend(loc='best', fontsize=19)
  #Save plot
  #multipagepdf.savefig(bbox_inches = 'tight')
  #plt.close()
  #print("Plotted Classifier Output to {}".format(options.plots))
  return plt

def compare_train_test(clf, ds_train, label_train, 
                      ds_test, label_test, mva='MVA', 
                      bins=50, use_vote=None, log=False):

  """
  Plots output of the classifier for the training and test dataset.

  Arguments:
  clf: classifier object.

  ds_train: dataset used for training.
  
  label_train: labels for the training datasets.
  
  ds_test: dataset used to test the classifier performance.
  
  label_test: labels of the corresponding test dataset.
  
  mva: Name of the ML method (used as axis label, str).
  
  bins: Number of bins (int).

  use_vote: Function to be able to use vote functions for folding classifiers (CAUTION: is not strictly correct).

  log: In order to use a logarithmic scale on the plots (boolean).
  """
  import numpy as np
  import matplotlib.pyplot as plt
  import inspect
  decisions = []
  ns_train = len(ds_train[label_train>0.5])
  nb_train = len(ds_train[label_train<0.5])

  for X,y in ((ds_train, label_train), (ds_test, label_test)):
    if use_vote == None:
      d1 = clf.predict_proba(X[y>0.5])[:,1]#.ravel()
      d2 = clf.predict_proba(X[y<0.5])[:,1]#.ravel()
    else:
      d1 = clf.predict_proba(X[y>0.5],vote_function=use_vote)[:,1]#.ravel()
      d2 = clf.predict_proba(X[y<0.5],vote_function=use_vote)[:,1]#.ravel()

    decisions += [d1, d2]
      
  low = min(np.min(d) for d in decisions)
  high = max(np.max(d) for d in decisions)
  low_high = (low,high)
  _, ax = plt.subplots()
  ys, bins, _ = plt.hist(decisions[0],
           color='r', alpha=0.5, range=low_high, bins=bins,
           histtype='step', normed=True,
           label='S (train)',log=log)
  yb, _, _ = plt.hist(decisions[1],
           color='b', alpha=0.5, range=low_high, bins=bins,
           histtype='step', normed=True,
           label='B (train)',log=log)
  width = (bins[1] - bins[0])
  
  minY = 1./float(max(ns_train,nb_train))/width
  maxY = 6*max(yb.max(),ys.max())
  
  if log==True:
    plt.ylim([minY,maxY])
  hist, bins = np.histogram(decisions[2],
                            bins=bins, range=low_high, normed=True)
  
  scale = len(decisions[2]) / sum(hist)
  err = np.sqrt(hist * scale) / scale

  width = (bins[1] - bins[0])
  center = (bins[:-1] + bins[1:]) / 2

  fillx = [bins[0]]
  fillys = [ys[0]]
  fillyb = [yb[0]]
  for i,(x,y_s,y_b) in enumerate(zip(bins,ys,yb)):
    if i==0:
      continue
    if i==len(yb)-1:
      fillx+=[x,x,bins[i+1]]
      fillyb+=[yb[i-1],y_b,y_b]
      fillys+=[ys[i-1],y_s,y_s]
      continue
    fillx+=[x,x]
    fillys+=[ys[i-1],y_s]
    fillyb+=[yb[i-1],y_b]

  ax.fill_between(fillx,minY,fillys, color='r', alpha=0.5)
  ax.fill_between(fillx,minY,fillyb, color='b', alpha=0.5)
  
  plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

  hist, bins = np.histogram(decisions[3],
                            bins=bins, range=low_high, normed=True)
  scale = len(decisions[3]) / sum(hist)
  err = np.sqrt(hist * scale) / scale

  plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

  plt.xlabel(mva+" output")
  plt.ylabel("Arbitrary units")
  plt.legend(loc='best')
  return plt


# ========================================
# ========================================

def plot_correlations(data,label='', **kwds):
    """
    Calculate pairwise correlation between features.
    
    Arguments:
    data: Pandas.DataFrame on which the correlations are calculated.

    label: prefix for the plot title and file name.

    Extra arguments are passed on to DataFrame.corr()
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pythonTools import ensure_dir
    # simply call df.corr() to get a table of
    # correlation values if you do not need
    # the fancy plotting
    corrmat = data.corr(**kwds)

    fig, ax1 = plt.subplots(ncols=1, figsize=(10,9))
    
    opts = {'cmap': plt.get_cmap("RdBu"),
            'vmin': -1, 'vmax': +1}
    heatmap1 = ax1.pcolor(corrmat, **opts)
    fig.colorbar(heatmap1, ax=ax1)


    ax1.set_title(label+" Correlations_{}vars".format(len(data.columns)-1))

    labels = corrmat.columns.values
    for ax in (ax1,):
        # shift location of ticks to center of the bins
        ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_xticklabels(labels, fontsize=12,minor=False, ha='right', rotation=70)
        ax.set_yticklabels(labels, fontsize=12,minor=False)
        
    fig.set_tight_layout(True)
    dir = 'plots'
    ensure_dir(dir)
    fig.savefig(dir+'/'+label+'_correlation_{}vars.pdf'.format(len(data.columns)))
    fig.savefig(dir+'/'+label+'_correlation_{}vars.png'.format(len(data.columns)))

