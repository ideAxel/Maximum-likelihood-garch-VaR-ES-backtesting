class Backtest:
    def __init__(self, model, returns):
        self.model = model
        self.returns = returns
        self.sigma2 = np.zeros(len(returns))
        self.sigma2[0] = np.var(returns)

    def run(self):
        estimated_parameters = self.model.optimize()
        print("Paramètres estimés:", estimated_parameters)

        # calculate VaR at 1% level
        var_1_percent = self.calculate_var(0.01)
        print("VaR à 1% : ", var_1_percent)

        # calculate ES at 1% level
        es_1_percent = self.calculate_es(0.01)
        print("ES à 1% : ", es_1_percent)

    def calculate_var(self, alpha):
        # calculate the VaR at a certain level
        standard_deviation = np.sqrt(self.sigma2[-1])  # the standard deviation of the latest return
        quantile = self.inv_normal_cdf(1-alpha)
        return -quantile * standard_deviation

    def calculate_es(self, alpha):
        # calculate the Expected Shortfall at a certain level
        var_alpha = self.calculate_var(alpha)
        returns_below_var = self.returns[self.returns <= -var_alpha]
        return -np.mean(returns_below_var) if len(returns_below_var) > 0 else np.nan

    def inv_normal_cdf(self, p):
        # inverse of the standard normal cumulative distribution function (CDF)
        # source: Peter J. Acklam, https://web.archive.org/web/20151028100006/http://home.online.no/~pjacklam/notes/invnorm/
        # constant values
        a1 = -39.69683028665376
        a2 = 220.9460984245205
        a3 = -275.9285104469687
        a4 = 138.3577518672690
        a5 = -30.66479806614716
        a6 = 2.506628277459239
        b1 = -54.47609879822406
        b2 = 161.5858368580409
        b3 = -155.6989798598866
        b4 = 66.80131188771972
        b5 = -13.28068155288572
        c1 = -7.784894002430293E-03
        c2 = -0.3223964580411365
        c3 = -2.400758277161838
        c4 = -2.549732539343734
        c5 = 4.374664141464968
        c6 = 2.938163982698783
        d1 = 7.784695709041462E-03
        d2 = 0.3224671290700398
        d3 = 2.445134137142996
        d4 = 3.754408661907416
        p_low = 0.02425
        p_high = 1 - p_low

        if p < p_low:
            q = np.sqrt(-2*np.log(p))
            return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1)
        elif p <= p_high:
            q = p - 0.5
            r = q*q
            return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)
        else:
            q = np.sqrt(-2*np.log(1-p))
            return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1)
