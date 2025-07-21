# xp

A framework for testing state-based CRDTs sync protocols.

## Usage

Run the binary using Cargo.
Either opt for run the simulation with GSets, AWSets or PNCounters.

```bash
$ cargo run -q -r -- awset
$ cargo run -q -r -- gset
$ cargo run -q -r -- pncounter
```

Such output can be fed into [`scrips/plots.py`](./scripts/plots.py)
to produce plots using matplotlib by using the following command:

```bash
$ python scripts/plots.py --help
```

`--output_data` produces an output that can be used to produce tables that display the raw experiment values in TeX format by using [`scripts/tables_transmitted.py`](./scripts/tables_transmitted.py).
The command is the following:

```bash
$ python scripts/tables_transmitted.py --help
```

`--output_ratios` produces an output that can be used to produce tables that display the metadata and redundancy ratios in TeX format by using [`scripts/tables_ratios.py`](./scripts/tables_ratios.py).
The command is the following:
```bash
$ python scripts/tables_ratios.py --help
```

> If you have any questions, [send me an email](mailto:pedromgomes29+github@gmail.com)
