import pandas as pd
import numpy as np
from utils.file_utils import absolute_path
from utils.config_utils import get_config

def create_txs_time_from_addresses():
    config = get_config('path')

    # Load necessary data
    addrTx = pd.read_csv(absolute_path(config['file']['addrTx']))
    txAddr = pd.read_csv(absolute_path(config['file']['txAddr']))
    txId2hash = pd.read_csv(absolute_path(config['file']['txId2hash']))
    txId2hash = txId2hash.rename(columns={'transaction': 'txhash'})
    wallets = pd.read_csv(absolute_path(config['file']['addr_features_classes']))  # or combined version

    # Normalize wallet features
    wallets = wallets[['address', 'first_block_appeared_in']].dropna()
    wallets['first_block_appeared_in'] = wallets['first_block_appeared_in'].astype(int)

    # Combine AddrTx and TxAddr
    addrTx = addrTx.rename(columns={'input_address': 'address'})[['txId', 'address']]
    txAddr = txAddr.rename(columns={'output_address': 'address'})[['txId', 'address']]

    all_tx_addr = pd.concat([addrTx, txAddr], axis=0)

    # Merge txId → address → block
    tx_blocks = pd.merge(all_tx_addr, wallets, on='address', how='left')

    # Aggregate: get minimum block number per transaction
    tx_blocks = tx_blocks.groupby('txId')['first_block_appeared_in'].min().reset_index()
    tx_blocks = tx_blocks.rename(columns={'first_block_appeared_in': 'block_number'})

    # Calculate block_timestamp using Bitcoin rules
    base_block = 391204
    base_time = 1448419200  # 2015-11-25 UTC
    avg_block_time = 600  # seconds

    tx_blocks['block_timestamp'] = tx_blocks['block_number'].apply(
        lambda b: int(base_time + (b - base_block) * avg_block_time)
    )

    # Merge with txId2hash to get txhash
    tx_blocks = pd.merge(tx_blocks, txId2hash, on='txId', how='inner')

    # Final format
    txs_time = tx_blocks[['txhash', 'block_timestamp', 'block_number']]
    txs_time = txs_time.dropna()
    txs_time.to_csv(absolute_path(config['file']['txs_time']), index=False)
    print(f"txs_time.csv created with {len(txs_time)} entries.")


if __name__ == '__main__':
    create_txs_time_from_addresses()