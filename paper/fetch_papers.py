#!/usr/bin/env python3
"""가이드 참고문헌의 arXiv 논문을 이 폴더로 내려받고, arXiv 원문 메타데이터로 제목을 교차검증한다.

- 제목/저자는 가이드 문서에서 사람이 옮겨 적은 값이므로 arXiv 원문과 대조해 오기를 잡는다.
  (실제로 2건 — LSQ, BEVFormer — 이 축약형이었고 이 대조로 잡혔다.)
- 메타데이터는 abs 페이지의 <meta name="citation_*"> 태그에서 긁는다.
  export.arxiv.org API를 먼저 썼지만 "Rate exceeded." (HTTP 429)로 레코드를 0건 주는 일이 잦았다.
- PDF는 매직바이트(%PDF)까지 확인해서 HTML 오류 페이지를 PDF로 착각하지 않게 한다.
- arXiv 권고에 따라 요청 사이에 3.2초를 쉰다. 전체 약 2분.
"""
import hashlib, html, json, pathlib, re, subprocess, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from papers import PAPERS

OUT = pathlib.Path(__file__).parent          # 이 스크립트가 있는 폴더에 받는다
META = pathlib.Path(__file__).parent / "paper_meta.json"
UA = "embedded-ai-quantization-guide/1.0 (study reference collector)"
PAUSE = 3.2                                  # arXiv 권고 간격


def curl(url, out=None, timeout=120):
    cmd = ["curl", "-sSL", "--max-time", str(timeout), "-A", UA,
           "-w", "%{http_code}"]
    if out:
        cmd += ["-o", str(out)]
    cmd.append(url)
    r = subprocess.run(cmd, capture_output=True, text=(out is not None))
    if out:
        return r.stdout.strip(), r.stderr
    body = r.stdout.decode("utf-8", "replace")
    return body[-3:], body[:-3]


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def abs_meta(aid):
    """abs 페이지의 citation_* 메타태그에서 제목/저자/등록일을 읽는다."""
    code, body = curl(f"https://arxiv.org/abs/{aid}")
    if code != "200":
        return None, f"HTTP {code}"
    get = lambda k: [html.unescape(m) for m in re.findall(
        rf'<meta[^>]+name="citation_{k}"[^>]+content="([^"]*)"', body)]
    title = get("title")
    if not title:
        return None, "citation_title 없음"
    return {"title": " ".join(title[0].split()),
            "authors": get("author"),
            "published": (get("date") or [""])[0][:10],
            "doi": (get("doi") or [""])[0],
            "journal": (get("journal_title") or [""])[0]}, ""


def main():
    print(f"논문 {len(PAPERS)}건 — PDF 다운로드 + 제목 대조\n")
    rows = []
    for i, (aid, cite, title, venue, group, stem, blurb, docs) in enumerate(PAPERS):
        rec = {"arxiv_id": aid, "cite": cite, "my_title": title, "venue": venue,
               "group": group, "stem": stem, "blurb": blurb, "docs": docs}

        # (1) 제목 교차검증 — arXiv 원문이 권위 있는 값이다
        m, why = abs_meta(aid)
        time.sleep(PAUSE)
        if m:
            rec.update(api_title=m["title"], authors=m["authors"],
                       published=m["published"], doi=m["doi"], journal=m["journal"])
            a, b = norm(title), norm(m["title"])
            rec["title_match"] = ("exact" if a == b else
                                  "shortened" if a and a in b else "differs")
        else:
            rec["title_match"] = "no-record"
            print(f"⚠️  {aid}: 메타데이터 못 읽음 ({why})")

        # (2) PDF 다운로드
        f = OUT / f"{aid}_{stem}.pdf"
        if f.exists() and f.stat().st_size > 20000:
            print(f"[{i+1:2d}/{len(PAPERS)}] {aid}  이미 있음 ({f.stat().st_size/1e6:.1f} MB)")
        else:
            code, err = curl(f"https://arxiv.org/pdf/{aid}", out=f)
            time.sleep(PAUSE)
            if code != "200":
                print(f"[{i+1:2d}/{len(PAPERS)}] {aid}  🔴 HTTP {code}")
                f.unlink(missing_ok=True)
                rec.update(pdf=None, ok=False, http=code)
                rows.append(rec); continue
        head = f.open("rb").read(5)
        if head[:4] != b"%PDF":
            print(f"[{i+1:2d}/{len(PAPERS)}] {aid}  🔴 PDF 아님 (head={head!r})")
            rec.update(pdf=None, ok=False, http="200-not-pdf")
            f.unlink(missing_ok=True)
            rows.append(rec); continue

        sz = f.stat().st_size
        md5 = hashlib.md5(f.read_bytes()).hexdigest()
        pages = len(re.findall(rb"/Type\s*/Page[^s]", f.read_bytes()))
        rec.update(pdf=f.name, ok=True, bytes=sz, md5=md5, pages=pages, http="200")
        print(f"[{i+1:2d}/{len(PAPERS)}] {aid}  ✅ {sz/1e6:5.1f} MB  ~{pages:3d}p  "
              f"제목대조={rec['title_match']}")
        rows.append(rec)

    META.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    ok = sum(r["ok"] for r in rows)
    print(f"\n다운로드 성공 {ok}/{len(rows)}   총 "
          f"{sum(r.get('bytes',0) for r in rows)/1e6:.1f} MB")
    diff = [(r["arxiv_id"], r["title_match"]) for r in rows
            if r["title_match"] != "exact"]
    print(f"제목 비-정확일치: {diff if diff else '없음 (전부 arXiv 원문과 동일)'}")


if __name__ == "__main__":
    main()
