#!/usr/bin/env python3
"""
CoreTally — HPC Recharge Summary MVP (Slurm 25.05 optimized)
- Actual-usage billing with fallbacks:
  * CPU-hours = CPUTimeRAW / 3600
  * GPU-hours = TRESUsageInAve['gres/gpu'] / 3600  (fallback: AllocTRES * ElapsedHours)
  * Mem GB-hours = AveRSS(GB) * ElapsedHours       (fallback: ReqMem(GB) * ElapsedHours)
- Stores monthly PI/account totals in SQLite
- FastAPI endpoints for JSON/CSV and PDF export

Expected Slurm configuration (25.05):
  slurm.conf
    AccountingStorageType=accounting_storage/slurmdbd
    AccountingStorageHost=<slurmdbd-host>
    SelectType=select/cons_tres
    SelectTypeParameters=CR_Core_Memory
    JobAcctGatherType=jobacct_gather/cgroup
    JobAcctGatherFrequency=30
  cgroup.conf
    CgroupAutomount=yes
    ConstrainCores=yes
    ConstrainRAMSpace=yes
    ConstrainDevices=yes
  gres.conf
    Name=gpu Type=<model> File=/dev/nvidia0 ... (and NodeName has Gres=gpu:<count>)

Usage:
  1) CLI:
     python coretally.py generate --month 2025-07
     python coretally.py csv --month 2025-07 --out july.csv
     python coretally.py pdf --month 2025-07 --out july.pdf
  2) API:
     uvicorn coretally:app --host 0.0.0.0 --port 8000
"""
import argparse, csv, datetime as dt, os, re, sqlite3, subprocess, sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Optional deps (API/PDF)
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, PlainTextResponse
except Exception:
    FastAPI = None

try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
except Exception:
    canvas = None

try:
    import yaml
except Exception:
    yaml = None

DB_PATH = os.environ.get("CORETALLY_DB", os.path.join(os.path.dirname(__file__), "usage.db"))
CONFIG_PATH = os.environ.get("CORETALLY_CONFIG", os.path.join(os.path.dirname(__file__), "config.yaml"))
PI_MAP_PATH = os.environ.get("CORETALLY_PI_MAP", os.path.join(os.path.dirname(__file__), "pi_accounts.csv"))
SACCT_BIN = os.environ.get("SACCT_BIN", "sacct")

DEFAULT_RATES = {"cpu_rate": 0.05, "gpu_rate": 0.50, "mem_rate": 0.01}

# Slurm 25.05 field set
SACCT_FIELDS = [
    "JobID","Account","User","AllocCPUS",
    "ElapsedRaw","CPUTimeRAW","State",
    "ReqMem","AveRSS","MaxRSS",
    "AllocTRES","TRESUsageInAve","TRESUsageInMax",
    "Start","End"
]

@dataclass
class UsageRow:
    account: str
    user: str
    alloc_cpus: int
    elapsed_hours: float
    cpu_time_hours: float
    gpu_hours_pref: float   # preferred GPU hours from TRESUsageInAve
    gpu_count_alloc: int    # fallback GPUs from AllocTRES
    mem_gb_req: float       # ReqMem in GB
    mem_gb_ave: float       # AveRSS in GB
    state: str

def load_config(path: str = CONFIG_PATH) -> dict:
    cfg = {"rates": DEFAULT_RATES, "sacct_extra_args": []}
    if yaml and os.path.exists(path):
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
            cfg["rates"] = data.get("rates", DEFAULT_RATES)
            cfg["sacct_extra_args"] = data.get("sacct_extra_args", [])
    return cfg

def ensure_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS usage_monthly (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pi_name TEXT NOT NULL,
        pi_email TEXT NOT NULL,
        account TEXT NOT NULL,
        month TEXT NOT NULL,
        cpu_hours REAL DEFAULT 0,
        gpu_hours REAL DEFAULT 0,
        mem_gb_hours REAL DEFAULT 0,
        job_count INTEGER DEFAULT 0,
        total_cost REAL DEFAULT 0
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_usage_monthly_month ON usage_monthly(month);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_usage_monthly_account ON usage_monthly(account);")
    con.commit(); con.close()

def parse_reqmem(reqmem: str, alloc_cpus: int) -> float:
    if not reqmem: return 0.0
    m = re.match(r"(?i)^\s*([0-9]+)([M|G])([c|n])?\s*$", reqmem.strip())
    if not m: return 0.0
    qty = float(m.group(1)); unit = m.group(2).upper(); scope = (m.group(3) or "n").lower()
    gb = qty/1024.0 if unit=="M" else qty
    if scope=="c": gb *= max(alloc_cpus,1)
    return gb

def rss_to_gb(rss_str: str) -> float:
    s = (rss_str or "").strip().upper()
    if not s: return 0.0
    if s.endswith("K"): return float(s[:-1]) / 1024 / 1024
    if s.endswith("M"): return float(s[:-1]) / 1024
    if s.endswith("G"): return float(s[:-1])
    # assume KB if raw
    try: return float(s) / 1024 / 1024
    except: return 0.0

def parse_alloc_tres_gpus(tres: str) -> int:
    if not tres: return 0
    total = 0
    for part in tres.split(","):
        if "=" not in part: continue
        key,val = part.split("=",1)
        if key.strip().lower().startswith("gres/gpu"):
            try: total += int(val)
            except: pass
    return total

def parse_tres_usage_seconds(tres_usage: str) -> Dict[str, float]:
    result = {}
    if not tres_usage: return result
    for part in tres_usage.split(","):
        if "=" not in part: continue
        key,val = part.split("=",1)
        key = key.strip().lower(); val = val.strip()
        if re.match(r"^\d+:\d{2}:\d{2}$", val):
            h,m,s = val.split(":"); secs = int(h)*3600 + int(m)*60 + int(s)
        else:
            try: secs = float(val)
            except: secs = 0.0
        result[key]=secs
    return result

def sacct_query(start_date: str, end_date: str, extra_args: List[str]) -> List[UsageRow]:
    cmd = [SACCT_BIN, "-a", "-X", "-n", "-P", "-S", start_date, "-E", end_date, "--format", ",".join(SACCT_FIELDS)] + extra_args
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    rows: List[UsageRow] = []
    for line in out.splitlines():
        cols = line.split("|")
        if len(cols)!=len(SACCT_FIELDS): continue
        (jobid, account, user, alloccpus,
         elapsedraw, cputimeraw, state,
         reqmem, averss, maxrss,
         alloctres, tresusageinave, tresusageinmax,
         start, end) = cols
        try:
            alloccpus = int(alloccpus or 0)
            elapsed_seconds = int(elapsedraw or 0); elapsed_hours = elapsed_seconds/3600.0
            cpu_time_hours = (int(cputimeraw or 0))/3600.0
            mem_gb_req = parse_reqmem(reqmem, alloccpus)
            mem_gb_ave = rss_to_gb(averss)
            gpu_count_alloc = parse_alloc_tres_gpus(alloctres)
            tres_ave = parse_tres_usage_seconds(tresusageinave)
            gpu_secs = tres_ave.get("gres/gpu", 0.0)
            gpu_hours_pref = float(gpu_secs)/3600.0 if gpu_secs else 0.0
        except Exception:
            continue
        rows.append(UsageRow(
            account=account or "unknown", user=user or "unknown",
            alloc_cpus=alloccpus, elapsed_hours=elapsed_hours,
            cpu_time_hours=cpu_time_hours, gpu_hours_pref=gpu_hours_pref,
            gpu_count_alloc=gpu_count_alloc, mem_gb_req=mem_gb_req,
            mem_gb_ave=mem_gb_ave, state=state or "UNKNOWN"
        ))
    return rows

def load_pi_map() -> Dict[str, Tuple[str, str]]:
    mapping = {}
    if os.path.exists(PI_MAP_PATH):
        with open(PI_MAP_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                acc = (r.get("account") or "").strip()
                name = (r.get("pi_name") or "").strip() or acc
                email = (r.get("pi_email") or "").strip() or f"{acc}@example.edu"
                if acc: mapping[acc]=(name,email)
    return mapping

def aggregate_month(month: str, cfg: dict, demo_csv: Optional[str] = None) -> List[dict]:
    start = dt.datetime.strptime(month + "-01", "%Y-%m-%d").date()
    end = dt.date(start.year + (1 if start.month==12 else 0), 1 if start.month==12 else start.month+1, 1)

    if demo_csv:
        raw_rows: List[UsageRow] = []
        with open(demo_csv, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                raw_rows.append(UsageRow(
                    account=r["account"], user=r["user"],
                    alloc_cpus=int(r["alloc_cpus"]), elapsed_hours=float(r["elapsed_hours"]),
                    cpu_time_hours=float(r["cpu_time_hours"]), gpu_hours_pref=float(r.get("gpu_hours_pref",0)),
                    gpu_count_alloc=int(r["gpu_count"]), mem_gb_req=float(r["mem_gb"]),
                    mem_gb_ave=float(r.get("mem_gb_ave",0)), state=r.get("state","COMPLETED")
                ))
    else:
        raw_rows = sacct_query(start.isoformat(), end.isoformat(), cfg.get("sacct_extra_args", []))

    by_account = {}
    for row in raw_rows:
        if row.state not in ("COMPLETED","FAILED","TIMEOUT"):
            continue
        acc = row.account or "unknown"
        d = by_account.setdefault(acc, {"cpu_hours":0.0,"gpu_hours":0.0,"mem_gb_hours":0.0,"job_count":0})
        d["cpu_hours"] += row.cpu_time_hours
        gpu_hours = row.gpu_hours_pref if row.gpu_hours_pref>0 else (row.gpu_count_alloc * row.elapsed_hours)
        d["gpu_hours"] += gpu_hours
        mem_gb = row.mem_gb_ave if row.mem_gb_ave>0 else row.mem_gb_req
        d["mem_gb_hours"] += mem_gb * row.elapsed_hours
        d["job_count"] += 1

    rates = cfg.get("rates", DEFAULT_RATES)
    pi_map = load_pi_map()

    summaries: List[dict] = []
    for account, m in by_account.items():
        pi_name, pi_email = pi_map.get(account, (account, f"{account}@example.edu"))
        total_cost = m["cpu_hours"]*rates["cpu_rate"] + m["gpu_hours"]*rates["gpu_rate"] + m["mem_gb_hours"]*rates["mem_rate"]
        summaries.append({
            "pi_name": pi_name, "pi_email": pi_email, "account": account, "month": month,
            "job_count": m["job_count"], "cpu_hours": m["cpu_hours"],
            "gpu_hours": m["gpu_hours"], "mem_gb_hours": m["mem_gb_hours"],
            "total_cost": round(total_cost,2)
        })
    return summaries

def save_to_db(summaries: List[dict]):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    for s in summaries:
        cur.execute("""INSERT INTO usage_monthly
        (pi_name, pi_email, account, month, cpu_hours, gpu_hours, mem_gb_hours, job_count, total_cost)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (s["pi_name"], s["pi_email"], s["account"], s["month"],
         s["cpu_hours"], s["gpu_hours"], s["mem_gb_hours"], s["job_count"], s["total_cost"]))
    con.commit(); con.close()

def generate_csv(summaries: List[dict]) -> str:
    import io
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=["pi_name","pi_email","account","month","job_count","cpu_hours","gpu_hours","mem_gb_hours","total_cost"])
    writer.writeheader()
    for s in summaries: writer.writerow(s)
    return out.getvalue()

def generate_pdf(summaries: List[dict], out_path: str):
    if not canvas: raise RuntimeError("reportlab not installed; cannot generate PDF")
    c = canvas.Canvas(out_path, pagesize=LETTER)
    width, height = LETTER; margin = 0.75*inch
    footer = ("CoreTally — Memory billed by AveRSS × elapsed (fallback ReqMem). "
              "GPU billed by TRESUsageInAve['gres/gpu'] (fallback AllocTRES × elapsed). "
              "CPU billed by CPUTimeRAW/3600. States: COMPLETED/FAILED/TIMEOUT.")

    for s in summaries:
        y = height - margin
        c.setFont("Helvetica-Bold", 14); c.drawString(margin, y, f"CoreTally Monthly Recharge Summary - {s['month']}"); y -= 20
        c.setFont("Helvetica", 11)
        c.drawString(margin, y, f"PI: {s['pi_name']}  |  Account: {s['account']}  |  Email: {s['pi_email']}"); y -= 16
        c.drawString(margin, y, f"Jobs: {s['job_count']}"); y -= 14
        c.drawString(margin, y, f"CPU Hours: {s['cpu_hours']:.2f}"); y -= 14
        c.drawString(margin, y, f"GPU Hours: {s['gpu_hours']:.2f}"); y -= 14
        c.drawString(margin, y, f"Memory GB-Hours: {s['mem_gb_hours']:.2f}"); y -= 20
        c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, f"Total Cost (USD): ${s['total_cost']:.2f}")
        c.setFont("Helvetica", 8); c.drawString(margin, 0.5*inch, footer)
        c.showPage()
    c.save()

# ------------- FastAPI -------------
if FastAPI:
    app = FastAPI(title="CoreTally API", version="0.2.0")
    @app.get("/summary/{month}")
    def get_month(month: str, format: str="json"):
        ensure_db()
        con = sqlite3.connect(DB_PATH); cur = con.cursor()
        cur.execute("""SELECT pi_name, pi_email, account, month, job_count, cpu_hours, gpu_hours, mem_gb_hours, total_cost
                       FROM usage_monthly WHERE month=? ORDER BY total_cost DESC""", (month,))
        rows = cur.fetchall(); con.close()
        cols=["pi_name","pi_email","account","month","job_count","cpu_hours","gpu_hours","mem_gb_hours","total_cost"]
        data=[dict(zip(cols,r)) for r in rows]
        if format=="csv": return PlainTextResponse(generate_csv(data), media_type="text/csv")
        return JSONResponse(data)

    @app.post("/summary/generate")
    def post_generate(month: Optional[str]=None, demo_csv: Optional[str]=None):
        if not month:
            today = dt.date.today().replace(day=1); prev = (today - dt.timedelta(days=1)).replace(day=1)
            month = prev.strftime("%Y-%m")
        cfg = load_config(); ensure_db()
        summaries = aggregate_month(month, cfg, demo_csv=demo_csv)
        save_to_db(summaries)
        return {"inserted": len(summaries), "month": month}

# ------------- CLI -------------
def main():
    parser = argparse.ArgumentParser(description="CoreTally — HPC Recharge Summary (25.05)")
    sub = parser.add_subparsers(dest="cmd")

    g = sub.add_parser("generate", help="Generate summaries for a month")
    g.add_argument("--month", help="YYYY-MM (defaults to prior month)")
    g.add_argument("--demo-csv", help="Path to demo CSV instead of running sacct")

    c = sub.add_parser("csv", help="Export CSV for a month from DB")
    c.add_argument("--month", required=True, help="YYYY-MM")
    c.add_argument("--out", default="recharge_summary.csv")

    p = sub.add_parser("pdf", help="Export PDF for a month from DB")
    p.add_argument("--month", required=True, help="YYYY-MM")
    p.add_argument("--out", default="recharge_summary.pdf")

    args = parser.parse_args()
    if args.cmd=="generate":
        month = args.month or (dt.date.today().replace(day=1) - dt.timedelta(days=1)).replace(day=1).strftime("%Y-%m")
        cfg = load_config(); ensure_db()
        summaries = aggregate_month(month, cfg, demo_csv=args.demo_csv)
        save_to_db(summaries); print(f"Inserted {len(summaries)} summaries for {month}")
    elif args.cmd=="csv":
        ensure_db(); con = sqlite3.connect(DB_PATH); cur = con.cursor()
        cur.execute("""SELECT pi_name, pi_email, account, month, job_count, cpu_hours, gpu_hours, mem_gb_hours, total_cost
                       FROM usage_monthly WHERE month=? ORDER BY total_cost DESC""", (args.month,))
        rows = cur.fetchall(); con.close()
        cols=["pi_name","pi_email","account","month","job_count","cpu_hours","gpu_hours","mem_gb_hours","total_cost"]
        data=[dict(zip(cols,r)) for r in rows]
        with open(args.out,"w",newline="") as f: f.write(generate_csv(data))
        print(f"Wrote {args.out}")
    elif args.cmd=="pdf":
        ensure_db(); con = sqlite3.connect(DB_PATH); cur = con.cursor()
        cur.execute("""SELECT pi_name, pi_email, account, month, job_count, cpu_hours, gpu_hours, mem_gb_hours, total_cost
                       FROM usage_monthly WHERE month=? ORDER BY total_cost DESC""", (args.month,))
        rows = cur.fetchall(); con.close()
        if not rows: print("No data"); sys.exit(1)
        cols=["pi_name","pi_email","account","month","job_count","cpu_hours","gpu_hours","mem_gb_hours","total_cost"]
        generate_pdf([dict(zip(cols,r)) for r in rows], args.out); print(f"Wrote {args.out}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
