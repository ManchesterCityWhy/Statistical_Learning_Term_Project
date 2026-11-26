cd ~/statlearn_real
python3 - <<'PY'
import re
s = open('common/core_metrics.py').read()
for name in ('pseudobulk', 'de_genes', '_signed_overlap', '_ranks', 'de_score', 'de_direction'):
    m = re.search(r'^def ' + name + r'\(.*?(?=^def |\Z)', s, flags=re.S | re.M)
    print(m.group(0)[:1600] if m else 'NOT FOUND ' + name)
    print('-----')
PY
