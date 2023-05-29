import numpy as np
from garch import GarchModel
from backtest import Backtest

def main():
    # Générer des données de retour aléatoires
    returns = np.random.randn(1000)

    # Instancier un modèle GARCH avec p=1 et q=1
    model = GarchModel(returns, 1, 1)

    # Optimiser le modèle GARCH
    model.optimize()

    # Instancier un backtest avec les rendements et le modèle
    backtest = Backtest(returns, model)

    # Calculer la VaR et l'ES
    var_95 = backtest.calculate_var(0.05)
    es_95 = backtest.calculate_es(0.05)

    print("VaR 95%: ", var_95)
    print("ES 95%: ", es_95)

if __name__ == "__main__":
    main()
