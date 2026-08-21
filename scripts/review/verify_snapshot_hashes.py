"""R-재현 리뷰(A): mock 스냅샷 해시를 독립 계산해 SNAPSHOT.sha256과 대조."""
import hashlib, pathlib, re, sys

ROOT = pathlib.Path("data/mock")
ok = True
for snapdir in sorted(ROOT.iterdir()):
    snapfile = snapdir / "SNAPSHOT.sha256"
    if not snapfile.exists():
        continue
    print(f"\n=== {snapdir.name} ===")
    declared, digest_line = {}, None
    for line in snapfile.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            m = re.search(r"snapshot_digest\s+([0-9a-f]{64})", line)
            if m:
                digest_line = m.group(1)
            continue
        if not line.strip():
            continue
        h, name = line.split(None, 1)
        declared[name.strip()] = h

    for name, exp in declared.items():
        raw = (snapdir / name).read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        match = got == exp
        ok &= match
        print(f"  {'OK  ' if match else 'FAIL'} {name}: {got[:16]}... (declared {exp[:16]}...)")
        # CRLF 민감도 점검: 줄끝을 바꾸면 해시가 달라지는가
        if b"\r\n" in raw:
            print(f"       * 파일에 CRLF 포함 — 체크아웃 설정에 따라 해시가 흔들릴 수 있음")

    # snapshot_digest 재구성 시도 (여러 후보 규칙)
    if digest_line:
        cands = {}
        names = list(declared)
        concat_hashes = "".join(declared[n] for n in names)
        cands["concat(hex) sorted-as-listed"] = hashlib.sha256(concat_hashes.encode()).hexdigest()
        cands["concat(hex) name-sorted"] = hashlib.sha256(
            "".join(declared[n] for n in sorted(names)).encode()).hexdigest()
        body = "".join(f"{declared[n]}  {n}\n" for n in names)
        cands["lines(hash  name)"] = hashlib.sha256(body.encode()).hexdigest()
        cands["concat(bytes) as-listed"] = hashlib.sha256(
            b"".join((snapdir / n).read_bytes() for n in names)).hexdigest()
        hit = [k for k, v in cands.items() if v == digest_line]
        print(f"  snapshot_digest {digest_line[:16]}... → 재구성 "
              f"{'성공: ' + hit[0] if hit else 'FAIL (규칙 미상 — 문서화 필요)'}")
        if not hit:
            for k, v in cands.items():
                print(f"       시도 {k}: {v[:16]}...")

print(f"\n판정: {'전건 일치' if ok else '불일치 존재'}")
sys.exit(0 if ok else 1)
