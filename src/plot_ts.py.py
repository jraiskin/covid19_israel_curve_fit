from src import utils
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = utils.get_ts_data()
    il_sr = df.loc['Israel']
    il_sr_nozeros = utils.rm_zeros(il_sr)

    n_days = il_sr_nozeros.size
    x_days = list(range(n_days))

    l_guess = 0.65 * 9 * (10 ** 6)
    k_guess = 1.0
    x0_guess = n_days if n_days <= 60 else n_days / 2

    params_opt, _ = utils.estimate_sigmoid_params(
        x_days,
        il_sr_nozeros.values,
        [l_guess, k_guess, x0_guess]
    )

    opt_l, opt_k, opt_x0 = params_opt

    x_range = np.linspace(0, n_days + 3, 1000)  # opt_x0 * 2

    print(opt_l)

    plt.figure()
    plt.plot(range(n_days), il_sr_nozeros, label='Data', marker='o')
    plt.plot(x_range, utils.sigmoid(x_range, *params_opt), 'r-', ls='-', label="Sigmoid Fit")
    plt.legend()
    plt.show()
