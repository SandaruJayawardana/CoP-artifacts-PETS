def interpolate(x_l, x_r, y_l, y_r, eps):
    print("x_l ", x_l, ", x_r ", x_r, ", y_l ", y_l, ", y_r ", y_r, ", eps ", eps)
    return y_l + (y_r - y_l) * (eps - x_l) / (x_r - x_l)


def get_per_attribute_privacy_budget(eps, eps_dict = {}, privacy_leakage_dict = {}):

    y_l = eps_dict[0]
    x_l = privacy_leakage_dict[0]

    for index_x_r, x_r in enumerate(privacy_leakage_dict[1:]):
        i = index_x_r + 1
        y_r = eps_dict[i]

        if eps >= x_l and eps <= x_r:
            return interpolate(x_l, x_r, y_l, y_r, eps)
        
        x_l = x_r
        y_l = y_r

    if eps <= privacy_leakage_dict[0]:
        x_l = privacy_leakage_dict[0]
        x_r = privacy_leakage_dict[1]
        y_l = eps_dict[0]
        y_r = eps_dict[1]
        return interpolate(x_l, x_r, y_l, y_r, eps)
    
    if eps >= privacy_leakage_dict[-1]:
        x_l = privacy_leakage_dict[-2]
        x_r = privacy_leakage_dict[-1]
        y_l = eps_dict[-2]
        y_r = eps_dict[-1]
        return interpolate(x_l, x_r, y_l, y_r, eps)

    raise ValueError(f"Eps = {eps} not bounded by {privacy_leakage_dict}")

  