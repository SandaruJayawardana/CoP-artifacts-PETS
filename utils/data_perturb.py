import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


def data_generator(parameter):
    data_values, eps, mechanism = parameter
    output_list = []
    
    for i in range(len(data_values)):
        if i % 5000 == 0:
            print("i", i)
        output_list.append(mechanism.gen_random_output(actual_value=list(data_values[i]), eps=eps))

    return output_list, eps

class Perturbation():
    def __init__(self, data, mechanism, COLUMNS, save_location):
        self.data = data
        self.COLUMNS = COLUMNS
        self.save_location = save_location
        self.mechanism = mechanism
        print("self.data[self.COLUMNS].values", self.data[self.COLUMNS].values)
    
    def perturb(self, EPS_ARRAY):
        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(data_generator, (self.data[self.COLUMNS].values, EPS_ARRAY[eps], self.mechanism)) for eps in range(len(EPS_ARRAY))]
            results_ = [future.result() for future in futures]

        for eps in EPS_ARRAY:
            for r in results_:
                if r[1] == eps:
                    final_perturbed_data_list = (np.array(r[0]))
                    print(np.shape(final_perturbed_data_list))
                    data_frame = pd.DataFrame(final_perturbed_data_list, columns=self.COLUMNS)
                    data_frame.to_csv(f"{self.save_location}_perturbed_eps_{eps}.csv.zip", compression='zip', index=False)
                    print("writing to ", f"{self.save_location}_perturbed_eps_{eps}.csv.zip")
                    break
        print('Perturbation Completed.')