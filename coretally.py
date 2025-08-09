#!/usr/bin/env python3
"""
CoreTally with:
- Actual-usage billing (CPUTimeRAW, TRESUsageInAve['gres/gpu'], AveRSS) + fallbacks
- Grant code tagging via Slurm WCKEY or mapping file
- Optional OIDC login (CILogon) for API routes
- Mail-out command to email Finance CSV + per-PI PDFs

Docs: see README.md
"""
import argparse, csv, datetime as dt, os, re, sqlite3, subprocess, sys, smtplib, tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from email.message import EmailMessage

# Optional deps (API/PDF/Auth)
try:
    from fastapi import FastAPI, HTTPException, Depends, Request
    from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
    from starlette.middleware.sessions import SessionMiddleware
except Exception:
    FastAPI = None
try:
    from authlib.integrations.starlette_client import OAuth
except Exception:
    OAuth = None

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
GRANT_MAP_PATH = os.environ.get("CORETALLY_GRANT_MAP", os.path.join(os.path.dirname(__file__), "grant_map.csv"))
SACCT_BIN = os.environ.get("SACCT_BIN", "sacct")

DEFAULT_RATES = {"cpu_rate": 0.05, "gpu_rate": 0.50, "mem_rate": 0.01}

# Slurm 25.05 field set
SACCT_FIELDS = [
    "JobID","Account","User","AllocCPUS",
    "ElapsedRaw","CPUTimeRAW","State",
    "ReqMem","AveRSS","MaxRSS",
    "AllocTRES","TRESUsageInAve","TRESUsageInMax",
    "WcKey","Cluster","Start","End"
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
    wckey: str
    cluster: str

def load_config(path: str = CONFIG_PATH) -> dict:
    cfg = {"rates": DEFAULT_RATES, "sacct_extra_args": [], "smtp": {}, "oidc": {}}
    if yaml and os.path.exists(path):
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
            cfg["rates"] = data.get("rates", DEFAULT_RATES)
            cfg["sacct_extra_args"] = data.get("sacct_extra_args", [])
            cfg["smtp"] = data.get("smtp", {})
            cfg["oidc"] = data.get("oidc", {})
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
        cluster TEXT DEFAULT '',
        grant_code TEXT DEFAULT '',
        cpu_hours REAL DEFAULT 0,
        gpu_hours REAL DEFAULT 0,
        mem_gb_hours REAL DEFAULT 0,
        job_count INTEGER DEFAULT 0,
        total_cost REAL DEFAULT 0
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_usage_monthly_month ON usage_monthly(month);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_usage_monthly_account ON usage_monthly(account);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_usage_monthly_grant ON usage_monthly(grant_code);")
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
         wckey, cluster, start, end) = cols
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
            mem_gb_ave=mem_gb_ave, state=state or "UNKNOWN",
            wckey=wckey or "", cluster=cluster or ""
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

def load_grant_map() -> Dict[str, str]:
    """
    grant_map.csv format:
    key,value
    pi_smith,GRANT-ABC-123    # by account
    johndoe,GRANT-XYZ-999     # by user
    WCKEY-PROJ,GRANT-DEF-777  # by WCKEY
    """
    mapping = {}
    if os.path.exists(GRANT_MAP_PATH):
        with open(GRANT_MAP_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                k = (r.get("key") or "").strip()
                v = (r.get("value") or "").strip()
                if k and v: mapping[k]=v
    return mapping

def resolve_grant_code(row: UsageRow, grant_map: Dict[str,str]) -> str:
    # Priority: WCKEY > user > account
    if row.wckey and row.wckey in grant_map: return grant_map[row.wckey]
    if row.user in grant_map: return grant_map[row.user]
    if row.account in grant_map: return grant_map[row.account]
    # fallback to WCKEY literal if set
    if row.wckey: return row.wckey
    return ""

def aggregate_month(month: str, cfg: dict, demo_csv: Optional[str] = None) -> List[dict]:
    start = dt.datetime.strptime(month + "-01", "%Y-%m-%d").date()
    end = dt.date(start.year + (1 if start.month==12 else 0), 1 if start.month==12 else start.month+1, 1)

    if demo_csv:
        # expect extended fields for demo: add wckey, cluster
        raw_rows: List[UsageRow] = []
        with open(demo_csv, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                raw_rows.append(UsageRow(
                    account=r["account"], user=r["user"],
                    alloc_cpus=int(r["alloc_cpus"]), elapsed_hours=float(r["elapsed_hours"]),
                    cpu_time_hours=float(r["cpu_time_hours"]), gpu_hours_pref=float(r.get("gpu_hours_pref",0)),
                    gpu_count_alloc=int(r["gpu_count"]), mem_gb_req=float(r["mem_gb"]),
                    mem_gb_ave=float(r.get("mem_gb_ave",0)), state=r.get("state","COMPLETED"),
                    wckey=r.get("wckey",""), cluster=r.get("cluster","")
                ))
    else:
        raw_rows = sacct_query(start.isoformat(), end.isoformat(), cfg.get("sacct_extra_args", []))

    grant_map = load_grant_map()
    by_key = {}
    for row in raw_rows:
        if row.state not in ("COMPLETED","FAILED","TIMEOUT"):
            continue
        key = (row.account or "unknown", resolve_grant_code(row, grant_map), row.cluster or "")
        d = by_key.setdefault(key, {"cpu_hours":0.0,"gpu_hours":0.0,"mem_gb_hours":0.0,"job_count":0})
        d["cpu_hours"] += row.cpu_time_hours
        gpu_hours = row.gpu_hours_pref if row.gpu_hours_pref>0 else (row.gpu_count_alloc * row.elapsed_hours)
        d["gpu_hours"] += gpu_hours
        mem_gb = row.mem_gb_ave if row.mem_gb_ave>0 else row.mem_gb_req
        d["mem_gb_hours"] += mem_gb * row.elapsed_hours
        d["job_count"] += 1

    rates = cfg.get("rates", DEFAULT_RATES)
    pi_map = load_pi_map()

    summaries: List[dict] = []
    for (account, grant_code, cluster), m in by_key.items():
        pi_name, pi_email = pi_map.get(account, (account, f"{account}@example.edu"))
        total_cost = m["cpu_hours"]*rates["cpu_rate"] + m["gpu_hours"]*rates["gpu_rate"] + m["mem_gb_hours"]*rates["mem_rate"]
        summaries.append({
            "pi_name": pi_name, "pi_email": pi_email, "account": account, "month": month,
            "cluster": cluster, "grant_code": grant_code,
            "job_count": m["job_count"], "cpu_hours": m["cpu_hours"],
            "gpu_hours": m["gpu_hours"], "mem_gb_hours": m["mem_gb_hours"],
            "total_cost": round(total_cost,2)
        })
    return summaries

def save_to_db(summaries: List[dict]):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    for s in summaries:
        cur.execute("""INSERT INTO usage_monthly
        (pi_name, pi_email, account, month, cluster, grant_code, cpu_hours, gpu_hours, mem_gb_hours, job_count, total_cost)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (s["pi_name"], s["pi_email"], s["account"], s["month"], s["cluster"], s["grant_code"],
         s["cpu_hours"], s["gpu_hours"], s["mem_gb_hours"], s["job_count"], s["total_cost"]))
    con.commit(); con.close()

def generate_csv(summaries: List[dict]) -> str:
    import io
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=[
        "pi_name","pi_email","account","month","cluster","grant_code",
        "job_count","cpu_hours","gpu_hours","mem_gb_hours","total_cost"
    ])
    writer.writeheader()
    for s in summaries: writer.writerow(s)
    return out.getvalue()

def generate_pdf(summaries: List[dict], out_path: str):
    if not canvas: raise RuntimeError("reportlab not installed; cannot generate PDF")
    c = canvas.Canvas(out_path, pagesize=LETTER)
    width, height = LETTER; margin = 0.75*inch
    footer = ("CoreTally — AveRSS × elapsed (fallback ReqMem); TRESUsageInAve['gres/gpu'] (fallback AllocTRES × elapsed); "
              "CPUTimeRAW/3600. States: COMPLETED/FAILED/TIMEOUT.")

    for s in summaries:
        y = height - margin
        c.setFont("Helvetica-Bold", 14); c.drawString(margin, y, f"CoreTally Monthly Recharge Summary - {s['month']}"); y -= 20
        c.setFont("Helvetica", 11)
        c.drawString(margin, y, f"PI: {s['pi_name']}  |  Account: {s['account']}  |  Email: {s['pi_email']}"); y -= 16
        if s.get("cluster"): c.drawString(margin, y, f"Cluster: {s['cluster']}"); y -= 14
        if s.get("grant_code"): c.drawString(margin, y, f"Grant Code: {s['grant_code']}"); y -= 14
        c.drawString(margin, y, f"Jobs: {s['job_count']}"); y -= 14
        c.drawString(margin, y, f"CPU Hours: {s['cpu_hours']:.2f}"); y -= 14
        c.drawString(margin, y, f"GPU Hours: {s['gpu_hours']:.2f}"); y -= 14
        c.drawString(margin, y, f"Memory GB-Hours: {s['mem_gb_hours']:.2f}"); y -= 20
        c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, f"Total Cost (USD): ${s['total_cost']:.2f}")
        c.setFont("Helvetica", 8); c.drawString(margin, 0.5*inch, footer)
        c.showPage()
    c.save()

# --------- Email utilities ---------
def send_email(smtp_cfg: dict, subject: str, body: str, to_addrs: List[str], attachments: List[Tuple[str, bytes]] = []):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_cfg.get("from_addr", smtp_cfg.get("username","noreply@example.edu"))
    msg["To"] = ", ".join(to_addrs)
    msg.set_content(body)
    for filename, data in attachments:
        msg.add_attachment(data, maintype="application", subtype="octet-stream", filename=filename)
    host = smtp_cfg.get("host"); port = int(smtp_cfg.get("port", 587))
    username = smtp_cfg.get("username"); password = smtp_cfg.get("password")
    use_tls = smtp_cfg.get("starttls", True)
    with smtplib.SMTP(host, port) as server:
        if use_tls: server.starttls()
        if username and password:
            server.login(username, password)
        server.send_message(msg)

def mailout(month: str, cfg: dict):
    ensure_db()
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("""SELECT pi_name, pi_email, account, month, cluster, grant_code, job_count, cpu_hours, gpu_hours, mem_gb_hours, total_cost
                   FROM usage_monthly WHERE month=? ORDER BY account""", (month,))
    rows = cur.fetchall(); con.close()
    cols = ["pi_name","pi_email","account","month","cluster","grant_code","job_count","cpu_hours","gpu_hours","mem_gb_hours","total_cost"]
    summaries = [dict(zip(cols, r)) for r in rows]
    if not summaries:
        print("No data for month", month); return

    # Finance CSV
    csv_bytes = generate_csv(summaries).encode("utf-8")

    # Per-PI PDFs
    attachments_finance = [("CoreTally_{}_Finance.csv".format(month), csv_bytes)]
    smtp_cfg = cfg.get("smtp", {})
    finance_to = smtp_cfg.get("finance_to", [])
    if isinstance(finance_to, str): finance_to = [finance_to]

    # Send Finance CSV
    if finance_to:
        send_email(smtp_cfg, f"CoreTally Finance CSV - {month}", "Attached: monthly recharge CSV.", finance_to, attachments_finance)

    # Group by PI email
    by_email = {}
    for s in summaries:
        by_email.setdefault(s["pi_email"], []).append(s)

    for email, items in by_email.items():
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            generate_pdf(items, tf.name)
            with open(tf.name, "rb") as f:
                pdf_bytes = f.read()
        send_email(smtp_cfg, f"CoreTally PI Summary - {month}", "Attached: your monthly summary.", [email],
                   [("CoreTally_{}_{}.pdf".format(month, items[0]['account']), pdf_bytes)])

# ------------- Auth (OIDC via CILogon) -------------
def auth_enabled(cfg: dict) -> bool:
    o = cfg.get("oidc", {})
    return bool(o.get("client_id") and o.get("client_secret") and o.get("redirect_uri"))

def get_auth(cfg: dict):
    if not OAuth:
        return None
    oauth = OAuth()
    oauth.register(
        name="cilogon",
        server_metadata_url="https://cilogon.org/.well-known/openid-configuration",
        client_id=cfg["oidc"]["client_id"],
        client_secret=cfg["oidc"]["client_secret"],
        client_kwargs={"scope": "openid email profile"}
    )
    return oauth

def require_login(cfg: dict):
    def dependency(request: Request):
        if not auth_enabled(cfg):
            return True  # auth disabled
        user = request.session.get("user")
        if not user:
            raise HTTPException(status_code=401, detail="Not authenticated")
        # Optional whitelist
        allowed_domains = cfg["oidc"].get("allowed_domains", [])
        if allowed_domains:
            email = user.get("email","")
            if not any(email.endswith("@"+d) or email.split("@")[-1]==d for d in allowed_domains):
                raise HTTPException(status_code=403, detail="Forbidden domain")
        return True
    return dependency

# ------------- FastAPI -------------
if FastAPI:
    cfg_boot = load_config()
    app = FastAPI(title="CoreTally API", version="0.3.0")
    if auth_enabled(cfg_boot):
        app.add_middleware(SessionMiddleware, secret_key=os.environ.get("CORETALLY_SESSION_SECRET","change-me-please"))
        oauth = get_auth(cfg_boot)
    else:
        oauth = None

    @app.get("/login")
    async def login(request: Request):
        if not oauth: raise HTTPException(status_code=501, detail="OIDC not configured")
        redirect_uri = cfg_boot["oidc"]["redirect_uri"]
        return await oauth.cilogon.authorize_redirect(request, redirect_uri)

    @app.get("/callback")
    async def auth_callback(request: Request):
        if not oauth: raise HTTPException(status_code=501, detail="OIDC not configured")
        token = await oauth.cilogon.authorize_access_token(request)
        userinfo = token.get("userinfo") or {}
        request.session["user"] = {"email": userinfo.get("email",""), "name": userinfo.get("name","")}
        return RedirectResponse(url="/me")

    @app.get("/me")
    async def me(request: Request):
        return request.session.get("user") or {}

    @app.get("/summary/{month}")
    def get_month(month: str, format: str="json", _: bool = Depends(require_login(cfg_boot))):
        ensure_db()
        con = sqlite3.connect(DB_PATH); cur = con.cursor()
        cur.execute("""SELECT pi_name, pi_email, account, month, cluster, grant_code, job_count, cpu_hours, gpu_hours, mem_gb_hours, total_cost
                       FROM usage_monthly WHERE month=? ORDER BY total_cost DESC""", (month,))
        rows = cur.fetchall(); con.close()
        cols=["pi_name","pi_email","account","month","cluster","grant_code","job_count","cpu_hours","gpu_hours","mem_gb_hours","total_cost"]
        data=[dict(zip(cols,r)) for r in rows]
        if format=="csv": return PlainTextResponse(generate_csv(data), media_type="text/csv")
        return JSONResponse(data)

    @app.post("/summary/generate")
    def post_generate(month: Optional[str]=None, demo_csv: Optional[str]=None, _: bool = Depends(require_login(cfg_boot))):
        if not month:
            today = dt.date.today().replace(day=1); prev = (today - dt.timedelta(days=1)).replace(day=1)
            month = prev.strftime("%Y-%m")
        cfg = load_config(); ensure_db()
        summaries = aggregate_month(month, cfg, demo_csv=demo_csv)
        save_to_db(summaries)
        return {"inserted": len(summaries), "month": month}

# ------------- CLI -------------
def main():
    parser = argparse.ArgumentParser(description="CoreTally v2 — HPC Recharge Summary (25.05)")
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

    m = sub.add_parser("mailout", help="Email Finance CSV + per-PI PDFs")
    m.add_argument("--month", required=True, help="YYYY-MM")
    m.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.cmd=="generate":
        month = args.month or (dt.date.today().replace(day=1) - dt.timedelta(days=1)).replace(day=1).strftime("%Y-%m")
        cfg = load_config(); ensure_db()
        summaries = aggregate_month(month, cfg, demo_csv=args.demo_csv)
        save_to_db(summaries); print(f"Inserted {len(summaries)} summaries for {month}")
    elif args.cmd=="csv":
        ensure_db(); con = sqlite3.connect(DB_PATH); cur = con.cursor()
        cur.execute("""SELECT pi_name, pi_email, account, month, cluster, grant_code, job_count, cpu_hours, gpu_hours, mem_gb_hours, total_cost
                       FROM usage_monthly WHERE month=? ORDER BY total_cost DESC""", (args.month,))
        rows = cur.fetchall(); con.close()
        cols=["pi_name","pi_email","account","month","cluster","grant_code","job_count","cpu_hours","gpu_hours","mem_gb_hours","total_cost"]
        data=[dict(zip(cols,r)) for r in rows]
        with open(args.out,"w",newline="") as f: f.write(generate_csv(data))
        print(f"Wrote {args.out}")
    elif args.cmd=="pdf":
        ensure_db(); con = sqlite3.connect(DB_PATH); cur = con.cursor()
        cur.execute("""SELECT pi_name, pi_email, account, month, cluster, grant_code, job_count, cpu_hours, gpu_hours, mem_gb_hours, total_cost
                       FROM usage_monthly WHERE month=? ORDER BY total_cost DESC""", (args.month,))
        rows = cur.fetchall(); con.close()
        if not rows: print("No data"); sys.exit(1)
        cols=["pi_name","pi_email","account","month","cluster","grant_code","job_count","cpu_hours","gpu_hours","mem_gb_hours","total_cost"]
        generate_pdf([dict(zip(cols,r)) for r in rows], args.out); print(f"Wrote {args.out}")
    elif args.cmd=="mailout":
        cfg = load_config(); ensure_db()
        if args.__dict__.get("dry_run"):
            print("Dry-run mode: not sending. Configure smtp in config.yaml and rerun without --dry-run.")
            return
        mailout(args.month, cfg); print("Mail-out complete.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
