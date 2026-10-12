"""Cloud metadata only: no model downloads, private media or API key output."""
import json
import os
from pathlib import Path

if os.environ.get("GITHUB_ACTIONS") != "true":
    raise SystemExit("Run on CI only")
import qai_hub as hub

client = hub.Client(hub.ClientConfig(api_token=os.environ["QAI_HUB_API_TOKEN"]))


def compact(value, depth=0):
    if isinstance(value, list):
        return {"entries": len(value)}
    if isinstance(value, dict):
        return ({"keys": list(value)[:30]} if depth >= 4 else
                {k: compact(v, depth + 1) for k, v in list(value.items())[:30]})
    return value


found = 0
for summary in client.get_job_summaries(limit=20):
    if not summary.name.startswith("Recall_D3Nano_"):
        continue
    found += 1
    print(json.dumps({"id": summary.job_id, "name": summary.name,
                      "status": str(summary.status), "url": summary.url}), flush=True)
    if summary.status.success and summary.name.endswith("_parity"):
        job = client.get_job(summary.job_id)
        profile = job.download_profile()
        print("Hosted profile: " + json.dumps(compact(profile)), flush=True)
        outputs = job.download_output_data()
        print("Output contract: " + json.dumps({k: [list(x.shape) for x in v] for k, v in outputs.items()}), flush=True)
    if summary.status.failure:
        job = client.get_job(summary.job_id)
        logs = job.download_job_logs("/tmp/recall-qnn-inspect/" + summary.job_id)
        for path in logs:
            p = Path(path)
            if p.suffix in (".log", ".txt"):
                print(p.name + ":\n" + p.read_text(errors="replace")[-12000:])
print("Matching recent conversion jobs:", found)
