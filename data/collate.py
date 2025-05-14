import torch


def collate_fn(batch):
    batch = batch[0]    # wrapper
    pair_flag = batch[0].get("x1") is not None
    force_flag = batch[0].get("force0") is not None

    if pair_flag and force_flag:
        keys = ["atype", "edge_mask", "mask", "x0", "x1", "force0", "force1", "pot0", "pot1"]
        types = [torch.long] * 2 + [torch.bool] + [torch.float] * 6
        flattens = [True] * 3 + [False] * 7
    elif pair_flag:
        keys = ["atype", "edge_mask", "mask", "x0", "x1"]
        types = [torch.long] * 2 + [torch.bool] + [torch.float] * 2
        flattens = [True] * 3 + [False] * 3
    elif force_flag:
        keys = ["atype", "edge_mask", "mask", "x0", "force0", "pot0"]
        types = [torch.long] * 2 + [torch.bool] + [torch.float] * 3
        flattens = [True] * 3 + [False] * 3
    else:
        keys = ["atype", "edge_mask", "mask", "x0"]
        types = [torch.long] * 2 + [torch.bool] + [torch.float]
        flattens = [True] * 3 + [False] * 1
    res = {}
    for key, _type, flatten in zip(keys, types, flattens):
        val = []
        for item in batch:
            val.append(torch.tensor(item[key], dtype=_type))
        if flatten:
            res[key] = torch.cat(val, dim=0)
        else:
            res[key] = torch.vstack(val)

    abid = []
    i = 0
    for item in batch:
        abid.extend([i] * len(item["atype"]))
        i += 1
    res["abid"] = torch.tensor(abid, dtype=torch.long)  # (N,)

    # environment
    res["env"] = batch[0]["env"]

    return res
