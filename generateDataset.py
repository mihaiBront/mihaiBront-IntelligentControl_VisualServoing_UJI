import matplotlib
matplotlib.use('Agg')  # Must be before any other matplotlib imports

import argparse
from lib.DataStoring import RowData, DataStore
from lib.IBVSHandler import IBVSHandler
import logging as log

log.basicConfig(level=log.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate IBVS dataset for neural network training')
    parser.add_argument('--num-sequences', type=int, default=1, help='Number of sequences to generate')
    parser.add_argument('--max-iterations', type=int, default=200, help='Maximum iterations per sequence')
    parser.add_argument('--lambda-value', type=float, default=0.1, help='Gain for the IBVS controller')
    parser.add_argument('--output-dir', type=str, default='dataset', help='Directory to save the dataset')
    parser.add_argument('--visualize', action='store_true', help='Visualize a sample sequence')
    parser.add_argument('--prepare-data', action='store_true', help='Prepare data for neural network training')
    parser.add_argument('--load', action='store_true', help='Load existing dataset instead of generating a new one')
    parser.add_argument('--savebunch-size', type=int, default=100, help='Number of sequences to save at a time')
    args = parser.parse_args()
    
    ibvsHandler = IBVSHandler(max_iterations=args.max_iterations,
                                       lambda_value=args.lambda_value)
    listAllData: list[RowData] = []
    dataStore = DataStore(path=args.output_dir)
    
    for i in range(args.num_sequences):
        ibvsHandler.randomizePositions()
        listAllData.extend(ibvsHandler.runIBVS(iteration=i))
        
        if i % args.savebunch_size == 0:
            dataStore.append_rows(listAllData)
            log.info(f"Saved {args.savebunch_size} sequences to {args.output_dir} (from {i-args.savebunch_size} to {i})")
    
    log.info(f"Saving remaining {len(listAllData) - dataStore.lastStoredRowId} sequences to {args.output_dir}")
    dataStore.append_rows(listAllData)
    
    print(len(listAllData))     
        
        
        
        