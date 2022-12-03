import os
import sys
import tarfile
import io
from random import shuffle
from collections import OrderedDict
import subprocess

from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Sampler, Dataset


def create_df_from_parquets(path, 
                            min_width=128, 
                            min_height=128,
                            verbose=True, 
                            max_files=None):
    
    parquets = sorted([f for f in os.listdir(path) if f.endswith(".parquet")])
    if max_files is not None:
        parquets = parquets[:max_files]

    loaded_parquet_paths = []
    dfs = []
    for f in parquets:
        try: 
            parquet_path = os.path.join(path, f)
            df = pd.read_parquet(parquet_path)
            df["parquet_id"] = f.replace(".parquet", "")
            dfs.append(df)
            
            loaded_parquet_paths.append(parquet_path)
        except Exception:
            if verbose:
                print("Could not load", f)  # download not completed yet for these (should be 8 at most) 
                
    # extract parquets
    for p in tqdm(loaded_parquet_paths, disable=not verbose, desc="Unpacking .tar files"):
        folder_path = p.replace(".parquet", "")
        tar_path = p.replace(".parquet", ".tar")
        if not os.path.exists(folder_path):
            subprocess.run(["mkdir", folder_path])
            subprocess.run(["tar", "-C", folder_path, "-xf", tar_path])
                
    # concat and filter df
    df = pd.concat(dfs)
    del dfs
    df = df[df["status"] == "success"].drop(columns=["status"])
    df = df[df["height"] >= min_height]
    df = df[df["width"] >= min_width]
    
    # define img paths (after extracting tars)
    img_paths = [os.path.join(path, df["parquet_id"].iloc[i], df["key"].iloc[i] + ".jpg") for i in range(len(df))]
    df["img_path"] = img_paths
    return df
    

def dist_to_bucket(height, width, bucket_width, bucket_height):
    return abs(height - bucket_height) + abs(width - bucket_width)


def assign_to_buckets(df, bucket_step_size=64, max_width=1024, max_height=768, min_bucket_count=64):
    # define widths and heights we want to considers
    heights_to_test = (df.height // bucket_step_size).unique() * bucket_step_size
    widths_to_test = (df.width // bucket_step_size).unique() * bucket_step_size
    combos_to_test = []
    for w in widths_to_test:
        for h in heights_to_test:
            if w * h <= max_height * max_width:
                combos_to_test.append((w, h))
    combos_to_test = np.array(combos_to_test)

    # filter out width and height combinations that appear less than 64 times in dataset
    dists = []
    for combo in combos_to_test:
        dist = dist_to_bucket(df.height, df.width, combo[0], combo[1])
        dists.append(dist)
    closest = np.array(dists).argmin(0)
    idcs, counts = np.unique(closest, return_counts=True)
    keep_idcs = idcs[counts >= min_bucket_count]
    combos_to_test_2 = combos_to_test[keep_idcs]
    # define final bucketing
    final_dists = []
    for combo in combos_to_test_2:
        dist = dist_to_bucket(df.height, df.width, combo[0], combo[1])
        final_dists.append(dist)
    bucket_assignments = np.array(final_dists).argmin(0)
    bucket_idcs, counts = np.unique(bucket_assignments, return_counts=True)
    # define bucket df
    bucket_df = pd.DataFrame({"bucket": range(len(combos_to_test_2)),
                              "bucket_width": combos_to_test_2[:, 0],
                              "bucket_height": combos_to_test_2[:, 1],
                              "bucket_size": counts,
                             })

    # merge bucket df into main df and define additional stuff
    df["bucket"] = bucket_assignments
    merged_df = df.merge(bucket_df, how="left", on="bucket")
    return merged_df


class BucketDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
        self.to_tensor = torchvision.transforms.ToTensor()
        self.norm = torchvision.transforms.Normalize([0.5], [0.5])
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):        
        img = None
        invalid_idcs = []
            
        while img is None:
            row = self.df.iloc[i]
            try:
                # return torch img and tokenized caption
                # get caption and tokenize
                caption = str(row.caption)
                input_ids = self.tokenizer(caption, max_length=self.tokenizer.model_max_length,
                                           padding="do_not_pad", truncation=True).input_ids
                padded_tokens = self.tokenizer.pad({"input_ids": input_ids}, 
                                              padding="max_length",
                                              #padding=True, 
                                              return_tensors="pt",
                                              max_length=77)


                pil_img = Image.open(row.img_path).convert("RGB")
                img = self.to_tensor(pil_img)
            except OSError:
                print(f"Could not load {row.img_path}")
                invalid_idcs.append(i)
                # get different random idx from same bucket
                bucket_rows = self.df[(self.df.bucket_width == row.bucket_width) & (self.df.bucket_height == row.bucket_height)]
                bucket_idcs = [idx for idx in bucket_rows.index.to_numpy() if idx not in set(invalid_idcs)]
                i = np.random.choice(bucket_idcs, 1)[0]
                
        # resize...
        bucket_width, bucket_height = int(row.bucket_width), int(row.bucket_height)
        channels, height, width = img.shape
        
        bucket_dims = (bucket_height, bucket_width)
        if bucket_width / bucket_height == width / height:
            img = torchvision.transforms.functional.resize(img, bucket_dims)
        else:
            img = torchvision.transforms.functional.resize(img, int(max(bucket_dims)))
            img = torchvision.transforms.RandomCrop((bucket_dims))(img)
        img = self.norm(img)
        #Either fits the bucket resolution exactly if the aspect ratio happens to match
        #or it extends past the bucket resolution on one dimension while fitting it exactly on the other.
        #In the latter case, a random crop is applied.
        #return img, tokenized
        return {
            "pixel_values": img,
            "input_ids": padded_tokens.input_ids,
            "attention_mask": padded_tokens.attention_mask,
        }
    

class BucketBatchSampler(Sampler):
    def __init__(self, bucket_assignments, batch_size):
        # put ids into the buckets
        bucketed_idcs = []
        for idx in sorted(np.unique(bucket_assignments)):
            bucketed_idcs.append(np.where(bucket_assignments == idx)[0])

        self.batch_size = batch_size
        self.bucketed_idcs = bucketed_idcs
        self.bucket_ids = np.arange(len(bucketed_idcs))
        self.bucket_sizes = torch.tensor([len(b) for b in self.bucketed_idcs]).float()
        self.total_items = sum(self.bucket_sizes)
        self.num_batches = int(np.ceil(self.total_items / batch_size))
        
        self.idcs = np.arange(len(bucketed_idcs))
        
        self.sampler = self

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            bucket_idx = torch.multinomial(self.bucket_sizes, 1, replacement=False).item()
            bucket_idcs = self.bucketed_idcs[bucket_idx]
            batch_idcs = np.random.choice(bucket_idcs, self.batch_size)
            yield batch_idcs
            

class BucketSampler(Sampler):
    def __init__(self, bucket_assignments, batch_size):
        # put ids into the buckets
        bucketed_idcs = []
        for idx in sorted(np.unique(bucket_assignments)):
            bucketed_idcs.append(np.where(bucket_assignments == idx)[0])

        self.batch_size = batch_size
        self.bucketed_idcs = bucketed_idcs
        self.bucket_ids = np.arange(len(bucketed_idcs))
        self.bucket_sizes = torch.tensor([len(b) for b in self.bucketed_idcs]).float()
        self.total_items = sum(self.bucket_sizes)
        self.num_batches = int(np.ceil(self.total_items / batch_size))
        
        self.idcs = np.arange(len(bucketed_idcs))
        
        #self.sampler = self

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return self.num_batches
    
    def sample_bucket(self):
        bucket_idx = torch.multinomial(self.bucket_sizes, 1, replacement=False).item()
        return bucket_idx

    def __iter__(self):
        count = 0
        
        bucket_idx = self.sample_bucket()
        bucket_idcs = self.bucketed_idcs[bucket_idx]
        
        for _ in range(self.num_batches):
            count += 1

            batch_idx = np.random.choice(bucket_idcs, 1)[0]
            
            if count == self.batch_size:
                # choose new bucket
                bucket_idx = self.sample_bucket()
                bucket_idcs = self.bucketed_idcs[bucket_idx]
                
            yield batch_idx
    
    