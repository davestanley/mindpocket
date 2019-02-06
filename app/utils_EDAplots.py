


def plotbar_train_dev(myvar,Ntrain,Ndev,varname,xlabel='Article #'):
    """Old version with limited axis labels"""
    # Import fig stuff
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

    figure(num=None, figsize=(15, 4),facecolor='w', edgecolor='k')
    barlist = plt.bar(range(len(myvar)), myvar, align='center', alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel('{} per article'.format(varname))
    for i in range(Ntrain,Ntrain+Ndev):
        barlist[i].set_color('r')
    plt.show()

def plotbar_train_dev2(myvar,Ntrain,Ndev,ylabel='value',xlabel='Article #'):
    # Import fig stuff
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

    figure(num=None, figsize=(15, 4),facecolor='w', edgecolor='k')
    barlist = plt.bar(range(len(myvar)), myvar, align='center', alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i in range(Ntrain,Ntrain+Ndev):
        barlist[i].set_color('r')
    plt.show()

def plot_train_dev(myvar,Ntrain,Ndev,varname,xlabel='Article #'):
    # Import fig stuff
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

    figure(num=None, figsize=(15, 4),facecolor='w', edgecolor='k')
    barlist = plt.plot(range(len(myvar)), myvar,)
    plt.xlabel(xlabel)
    plt.ylabel('{} per article'.format(varname))
    for i in range(Ntrain,Ntrain+Ndev):
        barlist[i].set_color('r')
    plt.show()

def plothist_train_dev(myvar,Ntrain,Ndev,varname,ylabel='N Articles',devbins=30):
    """Old version with limited axis labels"""
    # Import fig stuff
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    import statistics

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False,figsize=(15, 4));
    ax1.hist(myvar[0:Ntrain-1], bins=30);  # arguments are passed to np.histogram
    ax1.set_title("Narticles={}, median={}, mean={}".format(str(Ntrain),'{0:.2f}'.format(statistics.median(myvar[0:Ntrain-1])),'{0:.2f}'.format(statistics.mean(myvar[0:Ntrain-1]))));
    ax1.set_ylabel('N Articles');
    ax1.set_xlabel('{}'.format(varname));

    ax2.hist(myvar[Ntrain:], bins=devbins);  # arguments are passed to np.histogram
    ax2.set_title("Narticles={}, median={}, mean={}".format(str(Ndev),'{0:.2f}'.format(statistics.median(myvar[Ntrain:])),'{0:.2f}'.format(statistics.mean(myvar[Ntrain:]))));
    ax2.set_xlabel('{}'.format(varname));
    return {'ax1': ax1, 'ax2':ax2}


def plothist_train_dev2(myvar,Ntrain,Ndev,xlabel='value',ylabel='N Articles',devbins=30):
    # Import fig stuff
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    import statistics

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False,figsize=(15, 4));
    ax1.hist(myvar[0:Ntrain-1], bins=30);  # arguments are passed to np.histogram
    ax1.set_title("Narticles={}, median={}, mean={}".format(str(Ntrain),'{0:.2f}'.format(statistics.median(myvar[0:Ntrain-1])),'{0:.2f}'.format(statistics.mean(myvar[0:Ntrain-1]))));
    ax1.set_ylabel('N Articles');
    ax1.set_xlabel(xlabel);

    ax2.hist(myvar[Ntrain:], bins=devbins);  # arguments are passed to np.histogram
    ax2.set_title("Narticles={}, median={}, mean={}".format(str(Ndev),'{0:.2f}'.format(statistics.median(myvar[Ntrain:])),'{0:.2f}'.format(statistics.mean(myvar[Ntrain:]))));
    ax2.set_xlabel(xlabel);
    return {'ax1': ax1, 'ax2':ax2}
