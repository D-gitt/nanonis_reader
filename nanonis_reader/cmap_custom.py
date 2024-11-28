def nanox():
    import numpy as np
    import matplotlib.colors as clr
    cmap = clr.LinearSegmentedColormap.from_list('nanox', \
                                                 np.array([(0, 0, 0), (115, 33, 0), (180, 119, 0), (255, 255, 255)])/255, N=256)
    cmap.set_over('r')
    cmap.set_under('b')

    return cmap

def bwr():
    import matplotlib.colors as clr
    return clr.LinearSegmentedColormap.from_list('bwr_custom', \
                                                 [(0, 0, 0.4), (0, 0, 1), (1, 1, 1), (1, 0, 0), (0.4, 0, 0)], N=256)

def conduction_band():
    import numpy as np
    # return '#2A59A0'
    return np.array([42, 89, 160])/255

def valence_band():
    import numpy as np
    # return '#DF2B2B'
    return np.array([223, 43, 43])/255