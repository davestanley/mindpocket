


def plotbar_train_dev(myvar,Ntrain,Ndev,varname,xlabel='Article #'):
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


def plothist_train_dev(myvar,Ntrain,Ndev,varname,ylabel='N Articles'):
    # Import fig stuff
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    import statistics

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False,figsize=(15, 4));
    ax1.hist(myvar[0:Ntrain-1], bins=30);  # arguments are passed to np.histogram
    ax1.set_title("Narticles={}, median={}, mean={}".format(str(Ntrain),'{0:.1f}'.format(statistics.median(myvar[0:Ntrain-1])),'{0:.1f}'.format(statistics.mean(myvar[0:Ntrain-1]))));
    ax1.set_ylabel('N Articles');
    ax1.set_xlabel('{}'.format(varname));

    ax2.hist(myvar[Ntrain:], bins=30);  # arguments are passed to np.histogram
    ax2.set_title("Narticles={}, median={}, mean={}".format(str(Ndev),'{0:.1f}'.format(statistics.median(myvar[Ntrain:])),'{0:.1f}'.format(statistics.mean(myvar[Ntrain:]))));
    ax2.set_xlabel('{}'.format(varname));
