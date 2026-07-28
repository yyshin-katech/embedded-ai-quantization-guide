#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render.py — study_guide/*.md 를 다크 테마 HTML 로 변환하는 자체 렌더러.
외부 의존성 없음(pandoc/markdown 불필요). 오프라인 안전.

사용:
  python3 render.py <study_guide 디렉토리>     # 전체 렌더
  python3 render.py <파일.md>                   # 단일 파일 (사이드 nav 는 디렉토리 스캔)
"""
import sys, os, re, html, hashlib, glob

# ---- 문서 순서 & 짧은 라벨(네비게이션용) -------------------------------------
ORDER = [
    ("README.md", "홈"),
    ("01_environment_setup.md", "0. 환경준비"),
    ("02_deployment_ladder.md", "0.5 사다리"),
    ("03_quantization_theory.md", "1. 양자화이론"),
    ("04_transformer_quantization.md", "2. Transformer"),
    ("05_tensorrt.md", "3. TensorRT"),
    ("06_multi_soc.md", "4. 멀티SoC"),
    ("07_infrastructure.md", "5. 인프라화"),
    ("08_capstone.md", "캡스톤"),
    ("09_roadmap.md", "12주 로드맵"),
    ("10_pitfalls.md", "함정 5"),
]

def slugify(text):
    t = re.sub(r"<[^>]+>", "", text)              # 태그 제거
    t = re.sub(r"[^\w가-힣 -]", "", t).strip().lower()
    t = re.sub(r"[\s]+", "-", t)
    return t or "sec"

def ckey(label):
    norm = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", label)).strip()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()[:10]

def rewrite_link(url):
    m = re.match(r"^(README|\d{2}_[\w-]+)\.md(#.*)?$", url)
    if m:
        return m.group(1) + ".html" + (m.group(2) or "")
    return url

# ---- 인라인 변환 -------------------------------------------------------------
def inline(text):
    codes = []
    def stash(m):
        codes.append("<code>" + html.escape(m.group(1)) + "</code>")
        return "\x00%d\x00" % (len(codes) - 1)
    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  lambda m: '<a href="%s">%s</a>' % (html.escape(rewrite_link(m.group(2)), quote=True), m.group(1)),
                  text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\*\w])\*(?=\S)([^*\n]+?)(?<=\S)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], text)
    return text

# ---- 블록 파서 ---------------------------------------------------------------
def convert(md):
    lines = md.split("\n")
    i, n = 0, len(lines)
    out = []
    toc = []
    title = None

    def is_table_sep(s):
        return bool(re.match(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$", s))

    while i < n:
        line = lines[i]

        # 공백
        if not line.strip():
            i += 1; continue

        # 펜스 코드 (여는 백틱 개수 이상으로만 닫힘 → 4-backtick 래핑 지원)
        m = re.match(r"^\s*(`{3,})\s*([\w+-]*)\s*$", line)
        if m:
            flen = len(m.group(1)); lang = m.group(2)
            i += 1; buf = []
            while i < n:
                mc = re.match(r"^\s*(`{3,})\s*$", lines[i])
                if mc and len(mc.group(1)) >= flen:
                    break
                buf.append(lines[i]); i += 1
            i += 1  # 닫는 펜스
            cls = (' class="lang-%s"' % lang) if lang else ""
            out.append("<pre><code%s>%s</code></pre>" % (cls, html.escape("\n".join(buf))))
            continue

        # 헤딩
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1)); txt = m.group(2).strip()
            htxt = inline(txt)
            if lvl == 1 and title is None:
                title = re.sub(r"<[^>]+>", "", htxt)
                out.append("<h1>%s</h1>" % htxt)
            else:
                sid = slugify(txt)
                if lvl in (2, 3):
                    toc.append((lvl, sid, re.sub(r"<[^>]+>", "", htxt)))
                out.append('<h%d id="%s">%s</h%d>' % (lvl, sid, htxt, lvl))
            i += 1; continue

        # 수평선
        if re.match(r"^\s*([-*_])(\s*\1){2,}\s*$", line):
            out.append("<hr>"); i += 1; continue

        # 표
        if "|" in line and i + 1 < n and is_table_sep(lines[i + 1]):
            def cells(s):
                s = s.strip()
                if s.startswith("|"): s = s[1:]
                if s.endswith("|"): s = s[:-1]
                return [c.strip() for c in s.split("|")]
            headers = cells(line)
            aligns = []
            for c in cells(lines[i + 1]):
                l, r = c.startswith(":"), c.endswith(":")
                aligns.append("center" if l and r else "right" if r else "left" if l else "")
            i += 2; rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(cells(lines[i])); i += 1
            t = ['<div class="tablewrap"><table>', "<thead><tr>"]
            for j, h in enumerate(headers):
                a = (' style="text-align:%s"' % aligns[j]) if j < len(aligns) and aligns[j] else ""
                t.append("<th%s>%s</th>" % (a, inline(h)))
            t.append("</tr></thead><tbody>")
            for row in rows:
                t.append("<tr>")
                for j in range(len(headers)):
                    cell = row[j] if j < len(row) else ""
                    a = (' style="text-align:%s"' % aligns[j]) if j < len(aligns) and aligns[j] else ""
                    t.append("<td%s>%s</td>" % (a, inline(cell)))
                t.append("</tr>")
            t.append("</tbody></table></div>")
            out.append("".join(t)); continue

        # 블록쿼트 (콜아웃)
        if re.match(r"^\s*>\s?", line):
            buf = []
            while i < n and re.match(r"^\s*>\s?", lines[i]):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i])); i += 1
            inner = " ".join(x for x in buf if x.strip())
            joined = "\n".join(buf)
            cls = "callout"
            if "🔴" in joined or "금지" in joined: cls += " bad"
            elif "⚠️" in joined or "주의" in joined: cls += " warn"
            elif "💡" in joined or "팁" in joined: cls += " tip"
            out.append('<blockquote class="%s">%s</blockquote>' % (cls, inline(inner)))
            continue

        # 리스트 (task/unordered/ordered, 들여쓰기 중첩)
        if re.match(r"^\s*([-*+]|\d+\.)\s+", line):
            items = []
            while i < n and re.match(r"^\s*([-*+]|\d+\.)\s+", lines[i]):
                raw = lines[i]
                indent = len(raw) - len(raw.lstrip(" "))
                mo = re.match(r"^\s*(\d+)\.\s+(.*)$", raw)
                if mo:
                    items.append((indent, "ol", None, mo.group(2))); i += 1; continue
                mt = re.match(r"^\s*[-*+]\s+\[([ xX])\]\s+(.*)$", raw)
                if mt:
                    items.append((indent, "task", mt.group(1).lower() == "x", mt.group(2))); i += 1; continue
                mu = re.match(r"^\s*[-*+]\s+(.*)$", raw)
                items.append((indent, "ul", None, mu.group(1))); i += 1
            out.append(render_list(items))
            continue

        # 문단
        buf = []
        while i < n and lines[i].strip() and not re.match(
                r"^\s*(#{1,6}\s|```|>|[-*+]\s|\d+\.\s|([-*_])(\s*\2){2,}\s*$)", lines[i]) \
                and not ("|" in lines[i] and i + 1 < n and is_table_sep(lines[i + 1])):
            buf.append(lines[i]); i += 1
        if buf:
            out.append("<p>%s</p>" % inline(" ".join(buf)))

    return title or "문서", toc, "\n".join(out)

def render_list(items):
    # 들여쓰기 기반 단순 중첩. 각 레벨의 타입(ul/ol)은 첫 항목 기준.
    html_parts = []
    stack = []  # (indent, tag)
    def close_to(indent):
        while stack and stack[-1][0] > indent:
            html_parts.append("</%s>" % stack.pop()[1])
    for indent, kind, checked, text in items:
        tag = "ol" if kind == "ol" else "ul"
        if not stack or indent > stack[-1][0]:
            html_parts.append("<%s>" % tag); stack.append((indent, tag))
        else:
            close_to(indent)
            if not stack:
                html_parts.append("<%s>" % tag); stack.append((indent, tag))
        if kind == "task":
            k = ckey(text)
            chk = " checked" if checked else ""
            html_parts.append(
                '<li class="task"><label><input type="checkbox" data-k="%s"%s> %s</label></li>'
                % (k, chk, inline(text)))
        else:
            html_parts.append("<li>%s</li>" % inline(text))
    while stack:
        html_parts.append("</%s>" % stack.pop()[1])
    return "".join(html_parts)

# ---- 템플릿 -----------------------------------------------------------------
CSS = """
:root{--bg:#0f1115;--card:#171a21;--line:#2a2f3a;--tx:#e6e8ec;--dim:#9aa3b2;--acc:#5aa9ff;--ok:#3ecf8e;--warn:#f0b429;--bad:#ff6b6b;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);font-family:-apple-system,"Pretendard","Apple SD Gothic Neo","Malgun Gothic",sans-serif;line-height:1.75;font-size:16px}
a{color:var(--acc);text-decoration:none}a:hover{text-decoration:underline}
.bar{position:sticky;top:0;background:rgba(15,17,21,.96);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);z-index:20}
.bar-in{max-width:1000px;margin:0 auto;padding:10px 20px;display:flex;align-items:center;gap:14px}
.bar b{font-size:.9rem;white-space:nowrap}
.track{flex:1;height:8px;background:#252a34;border-radius:99px;overflow:hidden;min-width:80px}
.fill{height:100%;width:0;background:linear-gradient(90deg,var(--acc),var(--ok));transition:width .3s}
.pct{font-size:.82rem;color:var(--dim);min-width:96px;text-align:right;white-space:nowrap}
.stagenav{border-bottom:1px solid var(--line);background:#12151b}
.stagenav-in{max-width:1000px;margin:0 auto;padding:8px 20px;display:flex;flex-wrap:wrap;gap:6px}
.stagenav a{font-size:.76rem;padding:3px 9px;border-radius:99px;background:#1e222b;color:var(--dim)}
.stagenav a.cur{background:var(--acc);color:#08111f;font-weight:600}
.wrap{max-width:1000px;margin:0 auto;padding:34px 24px 120px}
h1{font-size:1.9rem;margin:0 0 20px;letter-spacing:-.02em;border-bottom:2px solid var(--acc);padding-bottom:16px}
h2{font-size:1.35rem;margin:48px 0 14px;padding-left:12px;border-left:4px solid var(--acc);scroll-margin-top:110px}
h3{font-size:1.08rem;margin:26px 0 10px;color:var(--acc);scroll-margin-top:110px}
h4{font-size:1rem;margin:20px 0 8px;color:var(--tx)}
p,li{color:var(--tx)}
ul,ol{padding-left:22px}li{margin:6px 0}
li.task{list-style:none;margin-left:-22px}
li.task label{cursor:pointer;display:flex;gap:9px;align-items:flex-start}
li.task input{margin-top:6px;width:16px;height:16px;accent-color:var(--acc);cursor:pointer;flex-shrink:0}
li.task input:checked ~ *{color:var(--dim)}
code{background:#20242e;color:#ffd479;padding:2px 6px;border-radius:4px;font-family:"SFMono-Regular",Consolas,monospace;font-size:.86em}
pre{background:#0b0d11;border:1px solid var(--line);border-radius:8px;padding:15px;overflow-x:auto;font-size:.85rem;line-height:1.55}
pre code{background:none;color:#c8d3e0;padding:0}
.tablewrap{overflow-x:auto;margin:16px 0}
table{width:100%;border-collapse:collapse;font-size:.9rem}
th,td{border:1px solid var(--line);padding:9px 12px;text-align:left;vertical-align:top}
th{background:#1e222b;color:var(--acc);font-weight:600}
blockquote{margin:16px 0;padding:14px 18px;border-radius:8px;background:var(--card);border:1px solid var(--line);border-left:4px solid var(--acc)}
blockquote.tip{border-left-color:var(--ok)}
blockquote.warn{border-left-color:var(--warn)}
blockquote.bad{border-left-color:var(--bad)}
hr{border:none;border-top:1px solid var(--line);margin:32px 0}
.toc{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 18px;margin:8px 0 28px}
.toc summary{cursor:pointer;color:var(--acc);font-weight:600;font-size:.92rem}
.toc ul{margin:10px 0 2px;padding-left:18px}
.toc li{margin:3px 0;font-size:.9rem}
.toc .l3{padding-left:16px;font-size:.85rem}
.pager{display:flex;justify-content:space-between;gap:12px;margin:16px 0 0}
.pager a{flex:1;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;font-size:.9rem}
.pager a.next{text-align:right}
.pager .lbl{display:block;font-size:.72rem;color:var(--dim)}
footer{margin-top:60px;padding-top:18px;border-top:1px solid var(--line);color:var(--dim);font-size:.82rem}
@media print{body{background:#fff;color:#000}.bar,.stagenav,.pager{display:none}.card,blockquote{background:#f7f7f7;border:1px solid #ccc}pre{background:#f4f4f4}pre code,code{color:#000}h2,h3{color:#000}a{color:#000}th{background:#eee;color:#000}}
"""

JS = """
(function(){
  var KEY='embai:'+PAGE;
  var boxes=document.querySelectorAll('input[type=checkbox][data-k]');
  var fill=document.getElementById('fill'), pct=document.getElementById('pct');
  var saved={};try{saved=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
  function upd(){var d=0;boxes.forEach(function(b){if(b.checked)d++});
    var p=boxes.length?Math.round(d/boxes.length*100):0;
    if(fill)fill.style.width=p+'%';
    if(pct)pct.textContent=boxes.length?(d+' / '+boxes.length+'  ('+p+'%)'):'체크리스트 없음';}
  boxes.forEach(function(b){var k=b.getAttribute('data-k');
    if(saved[k])b.checked=true;
    b.addEventListener('change',function(){saved[k]=b.checked;
      try{localStorage.setItem(KEY,JSON.stringify(saved))}catch(e){};upd();});});
  upd();
})();
"""

def build_html(fname, title, toc, body, order_present):
    # 스테이지 네비
    chips = []
    for f, lbl in ORDER:
        if f not in order_present: continue
        cur = " cur" if f == fname else ""
        chips.append('<a class="%s" href="%s">%s</a>' % (cur.strip(), f[:-3] + ".html", lbl))
    stagenav = '<div class="stagenav"><div class="stagenav-in">%s</div></div>' % "".join(chips)

    # 목차
    tochtml = ""
    if toc:
        lis = []
        for lvl, sid, txt in toc:
            cls = ' class="l3"' if lvl == 3 else ""
            lis.append('<li%s><a href="#%s">%s</a></li>' % (cls, sid, html.escape(txt)))
        tochtml = '<details class="toc" open><summary>목차</summary><ul>%s</ul></details>' % "".join(lis)

    # 이전/다음
    seq = [f for f, _ in ORDER if f in order_present]
    pager = ""
    if fname in seq:
        idx = seq.index(fname)
        prev_a = next_a = '<span></span>'
        if idx > 0:
            pf = seq[idx - 1]; pl = dict(ORDER)[pf]
            prev_a = '<a class="prev" href="%s"><span class="lbl">← 이전</span>%s</a>' % (pf[:-3] + ".html", pl)
        if idx < len(seq) - 1:
            nf = seq[idx + 1]; nl = dict(ORDER)[nf]
            next_a = '<a class="next" href="%s"><span class="lbl">다음 →</span>%s</a>' % (nf[:-3] + ".html", nl)
        pager = '<div class="pager">%s%s</div>' % (prev_a, next_a)

    page = fname[:-3]
    doc = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title><style>__CSS__</style></head><body>
<div class="bar"><div class="bar-in"><b>__BRAND__</b>
<div class="track"><div class="fill" id="fill"></div></div>
<span class="pct" id="pct">0 / 0</span></div></div>
__STAGENAV__
<div class="wrap">__TOC____PAGER____BODY____PAGER__
<footer>임베디드 AI 양자화 학습 가이드 · 체크 상태는 브라우저에 저장됩니다(localStorage) · 인쇄/PDF: Ctrl/⌘+P</footer>
</div>
<script>var PAGE=__PAGEJS__;__JS__</script>
</body></html>"""
    repl = {
        "__TITLE__": html.escape(title),
        "__CSS__": CSS,
        "__BRAND__": html.escape(title[:40]),
        "__STAGENAV__": stagenav,
        "__TOC__": tochtml,
        "__PAGER__": pager,
        "__BODY__": body,
        "__PAGEJS__": '"%s"' % page,
        "__JS__": JS,
    }
    for k, v in repl.items():
        doc = doc.replace(k, v)
    return doc

# ---- 메인 -------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: render.py <dir|file.md>"); sys.exit(1)
    target = sys.argv[1]
    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "*.md")))
        outdir = target
    else:
        files = [target]
        outdir = os.path.dirname(target) or "."
    present = set(os.path.basename(f) for f in glob.glob(os.path.join(outdir, "*.md")))
    done = []
    for path in files:
        fname = os.path.basename(path)
        with open(path, encoding="utf-8") as fh:
            md = fh.read()
        title, toc, body = convert(md)
        htmlpage = build_html(fname, title, toc, body, present)
        outpath = os.path.join(outdir, fname[:-3] + ".html")
        with open(outpath, "w", encoding="utf-8") as fh:
            fh.write(htmlpage)
        done.append((os.path.basename(outpath), len(htmlpage)))
    print("렌더 완료: %d개" % len(done))
    for name, size in done:
        print("  %-34s %6.1f KB" % (name, size / 1024))

if __name__ == "__main__":
    main()
