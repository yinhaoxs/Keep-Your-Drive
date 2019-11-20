import numpy as np
import pandas as pd

# accept loss_ratio predicting dataset, return scoring(0->100)
# compute score
def score(data, num=100, sort='pred', premium='prem_base_com', validate=False):
    data = data.sort_values(sort).reset_index(drop=True)

    total_prem = data[premium].sum()
    cum_prem = np.array(np.cumsum(data[premium]))
    average_prem = total_prem/num
    prem = average_prem
    locate = []

    # calculate locate
    for index, score in enumerate(cum_prem):
        if score >= prem:
            locate.append(index-1)
            prem += average_prem

    # check dimension
    if len(locate) < num:
        locate.append(data.shape[0] - 1)

    # calculate values
    values = []
    for i in locate:
        values.append(float(data[sort][i]))
    values.insert(0, -float("inf"))
    values[-1] = float("inf")

    # validation
    #TODO

    return locate, values

