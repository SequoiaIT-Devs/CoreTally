# CoreTally — HPC Recharge Summary (Slurm 25.05)

**Billing method**
- CPU-hours = `CPUTimeRAW / 3600`
- GPU-hours = `TRESUsageInAve['gres/gpu'] / 3600` (fallback: `AllocTRES × ElapsedHours`)
- Memory GB-hours = `AveRSS (GB) × ElapsedHours` (fallback: `ReqMem (GB) × ElapsedHours`)
- Billable states: `COMPLETED, FAILED, TIMEOUT`

**Expected Slurm configuration**
```
# slurm.conf
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=<slurmdbd-host>
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
JobAcctGatherType=jobacct_gather/cgroup
JobAcctGatherFrequency=30

# cgroup.conf
CgroupAutomount=yes
ConstrainCores=yes
ConstrainRAMSpace=yes
ConstrainDevices=yes

# gres.conf
Name=gpu Type=<model> File=/dev/nvidia0
# ensure NodeName has Gres=gpu:<count> in slurm.conf
```

## Usage
```bash
python coretally.py generate --month 2025-07
python coretally.py csv --month 2025-07 --out july.csv
python coretally.py pdf --month 2025-07 --out july.pdf
uvicorn coretally:app --host 0.0.0.0 --port 8000
```
