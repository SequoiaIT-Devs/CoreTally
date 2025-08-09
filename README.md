# CoreTally HPC Recharge Summary
Supported Scheduler: Slurm 25.05

## What’s new
- **Grant code tagging** (WCKEY or mapping file)
- **Mail-out** command (Finance CSV + per-PI PDFs)
- **Optional OIDC (CILogon)** login for API

## Billing method
- CPU-hours = `CPUTimeRAW / 3600`
- GPU-hours = `TRESUsageInAve['gres/gpu'] / 3600` (fallback: `AllocTRES × ElapsedHours`)
- Memory GB-hours = `AveRSS (GB) × ElapsedHours` (fallback: `ReqMem (GB) × ElapsedHours`)
- Billable states: `COMPLETED, FAILED, TIMEOUT`

## Expected Slurm configuration
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

## Grant Code Tagging
- Preferred: set **WCKEY** on jobs (`--wckey GRANT-ABC-123`); CoreTally reads `WcKey`.
- Or provide `grant_map.csv` with `key,value` where key can be **WCKEY**, **account**, or **username**.

## Usage
```bash
python coretally.py generate --month 2025-07
python coretally.py csv --month 2025-07 --out july.csv
python coretally.py pdf --month 2025-07 --out july.pdf
python coretally.py mailout --month 2025-07           # sends Finance CSV + PI PDFs
uvicorn coretally:app --host 0.0.0.0 --port 8000
```

## OIDC (CILogon) Setup (optional)
1. Install extras: `pip install authlib starlette` (already in requirements).
2. Set `oidc.client_id`, `oidc.client_secret`, `oidc.redirect_uri` in `config.yaml`.
3. Set `CORETALLY_SESSION_SECRET` environment variable to a random string.
4. Visit `/login` → approve → you’ll be redirected and sessioned.
5. Protected routes: `/summary/*` and POST `/summary/generate` when OIDC is configured.

## SMTP
Fill `smtp` block in `config.yaml`. Use `--dry-run` first to verify.
