import argparse
import os
from typing import List

import h5py

MAX_N_PRIM_ENC = 50
MAX_N_SLICERS = 10

def get_data_list(f, split="train") -> List[str]:
    data_list = list(f["split"][split][...].astype(str))

    data_list_filtered = []
    for data_id in data_list:
        n_prims = f["data"][data_id[-12:]]["sketch"]["prim"].attrs["len"]
        n_slicers = f["data"][data_id[-12:]]["zone_graph"]["prim"].attrs[
            "len"
        ]

        if n_prims > MAX_N_PRIM_ENC:
            continue
        if n_slicers > MAX_N_SLICERS:
            continue
        data_list_filtered.append(data_id)

    print(f"Using {len(data_list_filtered)}/{len(data_list)} {split} data")
    print(
        f"Filtered with {MAX_N_PRIM_ENC} prims & {MAX_N_SLICERS} slicers"
    )
    data_list = data_list_filtered

    return data_list

def main(args):

    f = h5py.File(args.h5, "r")

    data_list = get_data_list(f)

    with open(args.output, 'w+') as output_file:
        for data in data_list:
            output_file.write(f'{data}\n')

        
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--h5", type=str, default="ultimate_data.h5")
    parser.add_argument("--output", type=str, default="data_list.txt")

    args = parser.parse_args()

    main(args)