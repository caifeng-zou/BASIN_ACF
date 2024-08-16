import matplotlib.pyplot as plt
import copy

def normalize_by_trace(x):
    x_max = x.abs().amax(dim=-1)
    return x / x_max.view(-1, 1)

def data_preprocessing(st, starttime, t0, t1):
    st.detrend(type='demean')
    st.taper(0.05).filter('bandpass', freqmin=0.1, freqmax=1, zerophase=True) 
    st.trim(starttime + t0, starttime + t1)
    data = st.data
    return data

def section_plot(record, xs, ts, title, scale=0.3, 
                 save=False, figname=None, t0=None, t1=None,
                 figsize=(20, 8), fillcolors=(plt.get_cmap("coolwarm")(1.0), plt.get_cmap("coolwarm")(0.0)), alpha=0.8):
    """
    record: (nstation, nt)
    """
    record = copy.deepcopy(record)  # or record will be modified

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(record.size(0)):
        x = xs[i] + record[i] * scale
        if fillcolors[0] != None:
            ax.fill_betweenx(ts, x, xs[i], where=(x>xs[i]), color=fillcolors[0])
        if fillcolors[1] != None:
            ax.fill_betweenx(ts, x, xs[i], where=(x<xs[i]), color=fillcolors[1])
        ax.plot(x, ts, 'k', linewidth=1, alpha=alpha)

    ax.set_xlabel("Along-profile distance (km)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Time (s)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(xs[0]-1, xs[-1]+1)
    if t0 is None:
        t0 = ts[0]
        
    if t1 is None: 
        t1 = ts[-1]

    ax.set_ylim(t0, t1)
    ax.invert_yaxis()
    ax.tick_params(labelsize=12, width=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_weight('bold')
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    if save:
        if figname is not None:
            plt.savefig(figname, dpi=600, bbox_inches='tight', format='png')
        else:
            plt.savefig(title+'.png', dpi=600, bbox_inches='tight', format='png')