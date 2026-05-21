import os
import os.path as osp

import pickle
from joblib import Parallel, delayed
from typing import Callable, List, Tuple, Optional

import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import pandas as pd
import numpy as np

from .utils import data_from_molecule
from .constants import get_valid_atoms, get_valid_bonds
from ..utils.chem import get_largest

try:
    from rdkit import Chem, RDLogger

    WITH_RDKIT = True
except ImportError as e:
    WITH_RDKIT = False

try:
    import lmdb

    WITH_LMDB = True
except ImportError:
    WITH_LMDB = False


class PubChem(Dataset):
    """
    The **PubChem** dataset from the National Center for Biotechnology
    Information (NCBI), containing millions of small molecules represented
    by SMILES strings.

    Each molecule includes atom and bond features derived from its SMILES
    structure.

    Args:
        root (str, optional): Root directory where the dataset should be saved.
            (optional: :obj:`None`)
        transform (callable, optional): A function/transform that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            a :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before being saved to disk.
            (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            boolean value, indicating whether the data object should be
            included in the final dataset. (default: :obj:`None`)
        log (bool, optional): Whether to print any console output while
            downloading and processing the dataset. (default: :obj:`True`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        subset (bool): If True, uses a smaller subset for dev/debugging (20 times
            the value of `max_mols` in order to ensure max_mols will be reached).
            (default: :obj:`False`)
        max_mols (int, optional): Limit total number of molecules loaded.
            (default: :obj:`None`)
        variant (str, optional): Variant of the dataset to use (allows for multiple
            variants without overwriting).
            (default: :obj:`None`)
        num_workers (int, optional): Number of workers used for parallel
            processing of the dataset (default :obj:`os.cpu_count()-2`)
        drop_fragments (bool, optional): If True, detects the disconnected fragments
            from a molecule and only keeps the largest connected fragment in terms of
            heavy atoms count.
            (default: :obj:`True`)

    NOTE: This implementation uses LMDB storage.
    """

    # Raw PubChem CID–SMILES mapping file
    raw_url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"

    valid_atom_classes = get_valid_atoms("PubChem32")
    valid_bond_types = get_valid_bonds(with_aromatic=True)

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
        subset: bool = True,
        max_mols: Optional[int] = None,
        variant: str = None,
        num_workers: int = max(os.cpu_count() - 2, 1),
        drop_fragments: bool = True,
    ) -> None:
        # Require LMDB to be installed
        if not WITH_LMDB:
            raise ImportError(
                "LMDB is required for using this dataset. "
                "Install it with: 'pip install lmdb'"
            )

        self.subset = subset
        self.max_mols = max_mols
        self.variant = variant
        self.num_workers = num_workers
        self.drop_fragments = drop_fragments

        # LMDB environment
        self._env = None

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            log=log,
            force_reload=force_reload,
        )

    # ----------------------------------------------------------------------
    # Required properties
    # ----------------------------------------------------------------------

    @property
    def raw_file_names(self) -> List[str]:
        """
        The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading.
        """
        return ["PUBCHEM.csv", "CID-SMILES.gz"]

    @property
    def processed_dir(self) -> str:
        """
        The path of the directory where processed data of the dataset
        will be saved.
        """
        suffix = "" if self.variant is None else f"_{self.variant}"
        return osp.join(self.root, f"processed{suffix}")

    @property
    def processed_file_names(self) -> List[str]:
        """
        The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing.
        """
        return ["lmdb", "meta.pt"]

    # ----------------------------------------------------------------------
    # Dataset interface
    # ----------------------------------------------------------------------

    def len(self) -> int:
        """Returns the number of data objects stored in the dataset."""
        meta_path = self.processed_paths[1]
        if not osp.exists(meta_path):
            return 0
        meta = torch.load(meta_path)
        return meta["length"]

    def get(self, idx: int):
        """
        Gets the data object at index :obj:`idx`.

        Lazily open the LMDB environment to allow parallel access
        for `DataLoader` workers (each worker gets its own LMDB handle).
        """
        # Lazy initialization of LMDB environment for this worker/process
        if self._env is None:
            self._env = lmdb.open(
                self.processed_paths[0],
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1024,
            )

        # Retrieve data from LMDB
        with self._env.begin() as txn:
            data_bytes = txn.get(f"{idx}".encode("ascii"))
            if data_bytes is None:
                raise IndexError(f"Index {idx} not found in database")
            data = pickle.loads(data_bytes)

        return data

    # ----------------------------------------------------------------------
    # Download
    # ----------------------------------------------------------------------

    def download(self) -> None:
        # This code is based on the official GRALE repository: https://github.com/KrzakalaPaul/GRALE
        import gzip, shutil, requests

        csv_path = self.raw_paths[0]
        gz_path = self.raw_paths[1]

        if osp.exists(csv_path):
            print("PubChem raw CSV already exists, skipping download.")
            return

        os.makedirs(self.raw_dir, exist_ok=True)
        if self.log:
            print("Downloading PubChem CID–SMILES dataset")

        r = requests.get(self.raw_url)
        with open(gz_path, "wb") as f:
            f.write(r.content)

        with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)

        if self.log:
            print("Download complete.")

    # ----------------------------------------------------------------------
    # Processing
    # ----------------------------------------------------------------------

    def process(self) -> None:
        if not WITH_RDKIT:
            raise ImportError("RDKit is required for processing PubChem SMILES.")

        # Read PubChem CSV
        path = self.raw_paths[0]
        if self.log:
            print(f"Reading SMILES from {path}")
        df = pd.read_csv(path, sep="\t", header=None, names=["CID", "SMILES"])

        df_size = len(df)
        if self.log:
            print(f"Dataset size: {df_size:,} molecules")

        # Normalize maximum number of mols to be kept in processed dataset
        if self.max_mols is not None and self.max_mols > 0:
            self.max_mols = min(self.max_mols, df_size)
        else:
            self.max_mols = df_size

        # If `subset` is used, sample maximum of 20 * `max_mols`.
        # NOTE: The `20 * ` is arbitary and introduced to ensure that
        # `max_mols` can be reached, in case many molecules get dropped
        # due to filtering or processing errors.
        if self.subset:
            df_size = min(len(df), 20 * self.max_mols)
            df = df.sample(n=df_size, random_state=42)
            if self.log:
                print(f"Processing a sample of {df_size:,} molecules")

        if self.log:
            print(
                f"Maximum {self.max_mols:,} molecules will be kept after filtering and processing."
            )

        # Create LMDB environment
        ## Create directory in defined path
        db_path = self.processed_paths[0]
        os.makedirs(db_path, exist_ok=True)

        ## Estimate `map_size` needed for LMDB
        map_size = self._estimate_map_size(df, sample_size=300_000)
        if self.log:
            print(f"Creating LMDB database at {db_path}")

        env = lmdb.open(
            db_path,
            map_size=map_size,
            writemap=True,
            map_async=True,
            metasync=False,
        )

        # Process molecules and store in LMDB
        idx = 0
        seen_keys = set()
        batch_size = 20_000
        dropped = {k: 0 for k in ["invalid", "error", "prefilter", "duplicate"]}

        try:
            with tqdm(
                total=df_size,
                desc="Processing molecules",
                disable=not self.log,
                position=0,
            ) as pbar_dataset, tqdm(
                total=self.max_mols,
                desc="Valid molecules",
                disable=not self.log,
                position=1,
            ) as pbar_valid:
                # Process molecules in batches
                for batch_start in range(0, df_size, batch_size):
                    # Stop if we've reached max_mols
                    if idx >= self.max_mols:
                        break

                    # Get current batch of SMILES
                    batch_end = min(batch_start + batch_size, df_size)
                    batch_smiles = df["SMILES"].iloc[batch_start:batch_end].tolist()

                    # Process batch in parallel
                    results = Parallel(n_jobs=self.num_workers)(
                        delayed(_process_molecule_worker)(
                            smiles,
                            self.valid_atom_classes,
                            self.valid_bond_types,
                            self.drop_fragments,
                            self.pre_transform,
                            self.pre_filter,
                        )
                        for smiles in batch_smiles
                    )

                    # Collect successful results for batch writing
                    batch_data = []
                    for status, data in results:
                        if status == "success":
                            if idx >= self.max_mols:
                                break

                            # Deduplicate by SMILES
                            smiles = data.smiles.encode("utf-8")
                            if smiles in seen_keys:
                                dropped["duplicate"] += 1
                                continue
                            seen_keys.add(smiles)
                            batch_data.append((idx, data))
                            idx += 1
                        elif status in dropped:
                            dropped[status] += 1

                    # Write batch to LMDB
                    if batch_data:
                        self._write_batch(env, batch_data)

                    # Update dataset progress bar
                    pbar_dataset.update(len(batch_smiles))
                    pbar_valid.update(len(batch_data))

            # Force sync and close
            env.sync()
            env.close()

            # Save metadata
            meta = {
                "length": idx,
                "variant": self.variant,
                "subset": self.subset,
                "max_mols": self.max_mols,
                "dropped": dropped,
            }
            meta_path = self.processed_paths[1]
            torch.save(meta, meta_path)

        except Exception as e:
            import shutil

            env.close()
            if self.log:
                print(f"Processing failed: {e}. Cleaning up...")
                print(f"{idx:,} valid molecules were processed")
            if osp.exists(self.processed_paths[1]):
                os.unlink(self.processed_paths[1])
            shutil.rmtree(db_path, ignore_errors=True)
            raise

        else:
            # Log dataset stats
            if self.log:
                print("=" * 60)
                print(f"PubChem Variant '{self.variant}': {idx:,} valid molecules")
                print(f"Dropped molecules:")
                print(f"\tInvalid SMILES: {dropped['invalid']:,}")
                print(f"\tProcessing error: {dropped['error']:,}")
                print(f"\tPre-filter: {dropped['prefilter']:,}")
                print(f"\tDuplicates: {dropped['duplicate']:,}")
                print("=" * 60)
                print(f"Dataset saved to: {self.processed_dir}")

    def _write_batch(self, env, batch_data):
        """
        Write a batch of data to LMDB.
        """
        with env.begin(write=True) as txn:
            for idx, data in batch_data:
                # Serialize using pickle
                data_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                txn.put(f"{idx}".encode("ascii"), data_bytes)

    # ----------------------------------------------------------------------
    # Utility methods
    # ----------------------------------------------------------------------

    def close(self):
        """Close the LMDB environment. Call this when done with the dataset."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def __del__(self):
        """Cleanup LMDB environment on deletion."""
        self.close()

    def __repr__(self) -> str:
        meta_path = self.processed_paths[1]
        if osp.exists(meta_path):
            meta = torch.load(meta_path)
            variant_str = f" (variant: {self.variant})" if self.variant else ""
            return f"{self.__class__.__name__}{variant_str}({meta['length']})"
        return f"{self.__class__.__name__}(not processed)"

    def __getstate__(self):
        """Called before pickling (when forking workers). Close LMDB environment."""
        self.close()
        state = self.__dict__.copy()
        return state

    def _estimate_map_size(self, df: pd.DataFrame, sample_size: int) -> int:
        """
        Estimate total size needed for LMDB `map_size`.

        Args:
            df (pandas.DataFrame):
                The loaded dataframe containing the full dataset.
            sample_size (int):
                Number of samples to be used for average estimation.

        Returns:
            int: The estimated size in bytes.
        """
        sizes = []
        batch_size = 10_000

        # Shuffle indices to sample randomly
        indices = np.random.RandomState(42).permutation(len(df))

        with tqdm(
            total=sample_size, desc="Estimating LMDB size", disable=not self.log
        ) as pbar:
            # Process molecules in batches
            batch_start = 0
            while len(sizes) < sample_size and batch_start < len(df):
                # Get current batch of SMILES
                batch_end = min(batch_start + batch_size, len(df))
                batch_indices = indices[batch_start:batch_end]
                batch_smiles = df.iloc[batch_indices]["SMILES"].tolist()

                # Process batch in parallel using joblib
                results = Parallel(n_jobs=self.num_workers)(
                    delayed(_process_molecule_worker)(
                        smiles,
                        self.valid_atom_classes,
                        self.valid_bond_types,
                        self.drop_fragments,
                        self.pre_transform,
                        self.pre_filter,
                    )
                    for smiles in batch_smiles
                )

                # Collect sizes from valid molecules
                batch_valid = 0
                for status, data in results:
                    if status == "success" and data is not None:
                        if len(sizes) >= sample_size:
                            break

                        # Serialize like LMDB to estimate size
                        byte_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                        sizes.append(len(byte_data))
                        batch_valid += 1

                pbar.update(batch_valid)
                batch_start += batch_size

        if len(sizes) < sample_size:
            print(
                f"WARNING: Only {len(sizes):,} molecules were sampled "
                f"(target was {sample_size:,})"
            )

        # Calculate and optionally log stats
        sizes = np.array(sizes)
        safe_factor = 1.5
        avg_size = sizes.mean()
        max_size = sizes.max()
        percentile_95 = np.percentile(sizes, 95)
        expected_total = max_size * self.max_mols
        map_size = int(expected_total * safe_factor)

        if self.log:
            print(f"Dataset stats for {len(sizes)} samples")
            print(f"\tAverage size per molecule: {avg_size/1024:.2f} KB")
            print(f"\tMedian size: {np.median(sizes)/1024:.2f} KB")
            print(f"\t95th percentile: {percentile_95/1024:.2f} KB")
            print(f"\tMax size: {max_size/1024:.2f} KB")
            print(f"\tEstimated size: {expected_total/1024**3:.2f} GB")
            print(f"\tRecommended size ({safe_factor}x): {map_size/1024**3:.2f} GB")

        return map_size


def _process_molecule_worker(
    smiles: str,
    valid_atom_classes: List[int],
    valid_bond_types: List[Chem.BondType],
    drop_fragments: bool,
    pre_transform: Optional[Callable] = None,
    pre_filter: Optional[Callable] = None,
) -> Tuple[str, Optional[Data]]:
    """
    Process a single molecule from SMILES string in order to create the
    appropriate PyTorch Geometric `Data` representation.

    Args:
        smiles (str):
            SMILES representation of the molecule to be processed
        valid_atom_classes (List[int]):
            List with valid atom types (atomic numbers).
        valid_bond_types (List[rdkit.Chem.BondType]):
            List with valid bond types.
        drop_fragments (bool): If True, detects the disconnected fragments
            from a molecule and only keeps the largest connected fragment in terms of
            heavy atoms count.
        pre_transform (callable, optional): A function/transform that takes in
            a :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before being saved to disk.
            (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            boolean value, indicating whether the data object should be
            included in the final dataset. (default: :obj:`None`)

    Returns:
        Tuple[str, Optional[torch_geometric.data.Data]]:
            `(status, data)`:
            - `status` is either "success" or describing an error that
            occurred while processing the molecule ("invalid", "error", "prefilter").
                - *invalid* indicates incorrect SMILES string in dataset (not parsable
                by `RDKit`)
                - *error* indicates sanitization error, or existence of unknown atom
                types (see `valid_atom_classes`)
                - *prefilter* indicates that the molecule didn't pass through the
                given `pre_filter` function
            - `data` will be the `torch_geometric.data.Data` representation of the
            molecule in case of "success", otherwise `None`.

    NOTE: Defined outside the class so it can be picklable by workers (for
    parallel processing).
    """
    RDLogger.DisableLog("rdApp.*")

    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return "invalid", None
    if drop_fragments:
        mol = get_largest(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            return "error", None

    # Sanitize and process
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        # Chem.Kekulize(mol)
        # Convert to PyG data object
        data = data_from_molecule(
            molecule=mol,
            atom_classes=valid_atom_classes,
            bond_types=valid_bond_types,
        )
        if data is None:
            return "error", None

    except Exception:
        return "error", None

    # Apply pre-filter if available
    if pre_filter is not None and not pre_filter(data):
        return "prefilter", None

    # Apply pre-transform if available
    if pre_transform is not None:
        data = pre_transform(data)

    # Ensure that `smiles` exists on `data` object
    if getattr(data, "smiles", None) is None:
        data.smiles = smiles

    return "success", data
