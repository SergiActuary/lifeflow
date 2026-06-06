import polars as pl


def label(ids, grid, name="id"):
    if len(ids) != grid.shape[0]:
        raise ValueError(
            f"length mismatch: {len(ids)} ids but grid has {grid.shape[0]} rows"
        )
    if grid.ndim == 1:
        return pl.DataFrame({name: ids, "value": grid})
    return pl.DataFrame(
        {name: ids, **{f"t{j}": grid[:, j] for j in range(grid.shape[1])}}
    )
