from histogram_provider import HistogramProvider

if __name__ == '__main__':
    target = "imgs/target.jpg"
    source = "imgs/source.jpg"
    # TODO: D functionality
    d = None

    hist_provider = HistogramProvider(source, target)
    hist_provider.adap_hist_matching()
    hist_provider.plot()
